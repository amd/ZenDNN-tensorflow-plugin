/*******************************************************************************
 * Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *******************************************************************************/

#include <cmath>
#include <cstring>
#include <vector>

// TensorFlow plug-in headers.
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_kernel_common.h"
#include "tensorflow_plugin/src/amd_cpu/util/errors.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_kernel.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_requires.h"
#include "tensorflow_plugin/src/amd_cpu/util/register_types.h"
#include "tensorflow_plugin/src/amd_cpu/util/zen_utils.h"

// ZenDNNL Low Overhead API headers for embedding
#include "lowoha_operators/embedding_bag/lowoha_embag_common.hpp"
#include "lowoha_operators/embedding_bag/lowoha_embedding_bag.hpp"
// ZenDNNL logging support
#include "common/zendnnl_global.hpp"

namespace amd_cpu_plugin {

namespace {

// SafeCast post-processing: clamp bf16 infinities to zero.
inline void ClampBf16Infinities(Eigen::bfloat16* data, int64_t count) {
  for (int64_t i = 0; i < count; ++i) {
    if (std::isinf(static_cast<float>(data[i]))) {
      data[i] = Eigen::bfloat16(0.0f);
    }
  }
}

inline void ClampBf16Infinities(float*, int64_t) {
  // No-op for float output — SafeCast only needed for bf16.
}

}  // namespace

template <typename T_table, typename T_indices, typename T_output>
class ZenGroupEmbeddingOp : public OpKernel {
 public:
  ~ZenGroupEmbeddingOp() {}

  explicit ZenGroupEmbeddingOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_tables", &num_tables_));
    OP_REQUIRES_OK(context, context->GetAttr("N", &num_lookups_));
    OP_REQUIRES_OK(context, context->GetAttr("gather_axis", &gather_axis_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("gathers_per_table", &gathers_per_table_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("num_passthrough", &num_passthrough_));

    int sum = 0;
    for (int g : gathers_per_table_) sum += g;
    OP_REQUIRES(context, sum == num_lookups_,
                errors::InvalidArgument(
                    "_ZenGroupEmbedding: sum(gathers_per_table)=", sum,
                    " != N=", num_lookups_));
    OP_REQUIRES(
        context, static_cast<int>(gathers_per_table_.size()) == num_tables_,
        errors::InvalidArgument("_ZenGroupEmbedding: len(gathers_per_table)=",
                                gathers_per_table_.size(),
                                " != num_tables=", num_tables_));
    OP_REQUIRES(context, num_passthrough_ == 0 || num_passthrough_ == 1,
                errors::InvalidArgument(
                    "_ZenGroupEmbedding: num_passthrough must be 0 or 1, got ",
                    num_passthrough_));
  }

  void Compute(OpKernelContext* context) override {
    using namespace zendnnl::lowoha::embag;
    using namespace zendnnl::common;
    using namespace zendnnl::memory;

    // Inputs: tables[0..M-1], indices[M..M+N-1],
    //         passthrough[M+N] (if num_passthrough_==1)
    const int idx_input_base = num_tables_;
    const int pt_input_base = num_tables_ + num_lookups_;

    // Validate first table and determine axis.
    const Tensor& first_table = context->input(0);
    OP_REQUIRES(context, first_table.dims() >= 1,
                errors::InvalidArgument(
                    "_ZenGroupEmbedding: table must be at least 1-D, got ",
                    first_table.dims(), "-D"));

    const int table_rank = first_table.dims();
    int axis = gather_axis_;
    if (axis < 0) axis += table_rank;
    OP_REQUIRES(context, axis >= 0 && axis < table_rank,
                errors::InvalidArgument(
                    "_ZenGroupEmbedding: gather_axis ", gather_axis_,
                    " out of range for table rank ", table_rank));

    // Compute outer_size and inner_size from first table (must be the same
    // for all tables within a fused ConcatV2).
    int64_t outer_size = 1;
    for (int d = 0; d < axis; ++d) outer_size *= first_table.dim_size(d);
    int64_t inner_size = 1;
    for (int d = axis + 1; d < table_rank; ++d)
      inner_size *= first_table.dim_size(d);

    // Validate all tables have compatible outer_size and inner_size.
    for (int t = 1; t < num_tables_; ++t) {
      const Tensor& tbl = context->input(t);
      OP_REQUIRES(context, tbl.dims() == table_rank,
                  errors::InvalidArgument("_ZenGroupEmbedding: table ", t,
                                          " rank=", tbl.dims(),
                                          " != first table rank=", table_rank));
      int64_t t_outer = 1;
      for (int d = 0; d < axis; ++d) t_outer *= tbl.dim_size(d);
      int64_t t_inner = 1;
      for (int d = axis + 1; d < table_rank; ++d) t_inner *= tbl.dim_size(d);
      OP_REQUIRES(context, t_outer == outer_size && t_inner == inner_size,
                  errors::InvalidArgument(
                      "_ZenGroupEmbedding: table ", t, " outer_size=", t_outer,
                      " inner_size=", t_inner,
                      " incompatible with first table outer_size=", outer_size,
                      " inner_size=", inner_size));
    }

    const int64_t num_indices = context->input(idx_input_base).NumElements();
    for (int i = 1; i < num_lookups_; ++i) {
      OP_REQUIRES(
          context,
          context->input(idx_input_base + i).NumElements() == num_indices,
          errors::InvalidArgument(
              "_ZenGroupEmbedding: index tensor size mismatch: ",
              context->input(idx_input_base + i).NumElements(), " vs ",
              num_indices));
    }

    zendnnl::error_handling::apilog_info(
        "_ZenGroupEmbedding Compute: num_tables=", num_tables_,
        ", num_lookups=", num_lookups_, ", gather_axis=", gather_axis_,
        " (normalized=", axis, ")", ", outer_size=", outer_size,
        ", inner_size=", inner_size, ", num_indices=", num_indices);

    // Compute gather features per output row.
    const int64_t per_index_features = num_indices * inner_size;
    int64_t gather_row_features = 0;
    for (int g : gathers_per_table_) {
      gather_row_features += g * per_index_features;
    }

    // Passthrough: appended after gather results in each output row.
    int64_t pt_row_features = 0;
    if (num_passthrough_ == 1) {
      const Tensor& pt = context->input(pt_input_base);
      OP_REQUIRES(context, pt.dims() == table_rank,
                  errors::InvalidArgument(
                      "_ZenGroupEmbedding: passthrough rank=", pt.dims(),
                      " != table rank=", table_rank));
      int64_t pt_outer = 1;
      for (int d = 0; d < axis; ++d) pt_outer *= pt.dim_size(d);
      OP_REQUIRES(context, pt_outer == outer_size,
                  errors::InvalidArgument(
                      "_ZenGroupEmbedding: passthrough outer_size=", pt_outer,
                      " != gather outer_size=", outer_size));
      pt_row_features = 1;
      for (int d = axis; d < table_rank; ++d) pt_row_features *= pt.dim_size(d);
    }

    const int64_t output_row_features = gather_row_features + pt_row_features;
    TensorShape out_shape({outer_size, output_row_features});
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    if (output->NumElements() == 0) return;

    data_type_t table_dt = std::is_same<T_table, Eigen::bfloat16>::value
                               ? data_type_t::bf16
                               : data_type_t::f32;
    data_type_t output_dt = std::is_same<T_output, Eigen::bfloat16>::value
                                ? data_type_t::bf16
                                : data_type_t::f32;
    data_type_t indices_dt = std::is_same<T_indices, int32>::value
                                 ? data_type_t::s32
                                 : data_type_t::s64;

    T_output* out_data = output->flat<T_output>().data();

    // Threshold for choosing direct gather vs embedding_bag path.
    // For small feature counts (<16 features per index), the direct gather
    // path with simple memory copies is faster than the overhead of setting
    // up the embedding_bag infrastructure. This threshold was determined
    // empirically on AMD EPYC processors with typical embedding dimensions.
    // TODO (plugin): make this tunable via an environment variable or attribute
    // for different hardware configurations
    constexpr int64_t kDirectGatherThreshold = 16;
    // Choose between direct gather loop (small per-op work) and
    // group_embedding_bag_direct (large embedding vectors).
    const bool use_direct = (per_index_features < kDirectGatherThreshold);

    if (use_direct) {
      // Direct path: simple indexed copy with inline type conversion.
      // Avoids 102K+ embedding_bag ops when inner_size is small (e.g., 1).
      zendnnl::error_handling::apilog_info(
          "_ZenGroupEmbedding direct gather: num_lookups=", num_lookups_,
          ", outer_size=", outer_size,
          ", per_index_features=", per_index_features);

      int64_t col_offset = 0;
      int idx_cursor = 0;

      for (int t = 0; t < num_tables_; ++t) {
        const Tensor& tbl = context->input(t);
        const T_table* tbl_data = tbl.flat<T_table>().data();
        const int64_t gather_dim = tbl.dim_size(axis);
        const int64_t table_outer_stride = gather_dim * inner_size;
        const int num_gathers = gathers_per_table_[t];

        for (int g = 0; g < num_gathers; ++g) {
          const Tensor& idx_t = context->input(idx_input_base + idx_cursor + g);
          const T_indices* idx_data = idx_t.flat<T_indices>().data();

          for (int64_t ni = 0; ni < num_indices; ++ni) {
            const int64_t idx_val = static_cast<int64_t>(idx_data[ni]);
            OP_REQUIRES(
                context, idx_val >= 0 && idx_val < gather_dim,
                errors::InvalidArgument("_ZenGroupEmbedding: index ", idx_val,
                                        " out of bounds [0, ", gather_dim,
                                        ") for table ", t, " gather ", g));
            for (int64_t o = 0; o < outer_size; ++o) {
              const T_table* src =
                  tbl_data + o * table_outer_stride + idx_val * inner_size;
              T_output* dst = out_data + o * output_row_features + col_offset +
                              ni * inner_size;
              for (int64_t k = 0; k < inner_size; ++k) {
                dst[k] = static_cast<T_output>(src[k]);
              }
            }
          }
          col_offset += per_index_features;
        }
        idx_cursor += num_gathers;
      }

      // Passthrough: direct memcpy into the tail of each output row.
      if (num_passthrough_ == 1 && pt_row_features > 0) {
        const Tensor& pt = context->input(pt_input_base);
        const T_output* pt_data = pt.flat<T_output>().data();
        const int64_t pt_bytes = pt_row_features * sizeof(T_output);
        for (int64_t o = 0; o < outer_size; ++o) {
          std::memcpy(out_data + o * output_row_features + gather_row_features,
                      pt_data + o * pt_row_features, pt_bytes);
        }
      }
    } else {
      // Embedding bag path: efficient for large embedding_dim.
      const int64_t total_ops = outer_size * num_lookups_;

      std::vector<const void*> tables_vec(total_ops);
      std::vector<const void*> indices_ptrs(total_ops);
      std::vector<const void*> offsets_ptrs(total_ops, nullptr);
      std::vector<const float*> weights_ptrs(total_ops, nullptr);
      std::vector<void*> dst_ptrs(total_ops);
      std::vector<embag_params_t> params(total_ops);

      int64_t output_col_offset = 0;
      int idx_cursor = 0;
      int64_t op_cursor = 0;

      for (int t = 0; t < num_tables_; ++t) {
        const Tensor& tbl = context->input(t);
        const T_table* tbl_data = tbl.flat<T_table>().data();
        const int64_t gather_dim = tbl.dim_size(axis);
        const int64_t table_outer_stride = gather_dim * inner_size;
        const int num_gathers = gathers_per_table_[t];

        embag_params_t ep;
        ep.dtypes.table = table_dt;
        ep.dtypes.output = output_dt;
        ep.dtypes.indices = indices_dt;
        ep.algo = embag_algo_t::none;
        ep.num_embeddings = static_cast<uint64_t>(gather_dim);
        ep.embedding_dim = static_cast<uint64_t>(inner_size);
        ep.num_indices = static_cast<uint64_t>(num_indices);
        ep.num_bags = 0;
        ep.is_weights = false;
        ep.include_last_offset = false;
        ep.padding_idx = -1;

        for (int64_t o = 0; o < outer_size; ++o) {
          for (int g = 0; g < num_gathers; ++g) {
            const int64_t op_idx = op_cursor + o * num_gathers + g;
            const Tensor& idx = context->input(idx_input_base + idx_cursor + g);

            tables_vec[op_idx] = tbl_data + o * table_outer_stride;
            indices_ptrs[op_idx] = idx.flat<T_indices>().data();
            dst_ptrs[op_idx] = out_data + o * output_row_features +
                               output_col_offset + g * per_index_features;
            params[op_idx] = ep;
          }
        }

        output_col_offset += num_gathers * per_index_features;
        idx_cursor += num_gathers;
        op_cursor += outer_size * num_gathers;
      }

      zendnnl::error_handling::apilog_info(
          "_ZenGroupEmbedding embag dispatch: total_ops=", total_ops,
          ", output_shape=", out_shape.DebugString());

      status_t st =
          group_embedding_bag_direct(tables_vec, indices_ptrs, offsets_ptrs,
                                     weights_ptrs, dst_ptrs, params);

      OP_REQUIRES(context, st == status_t::success,
                  errors::Internal(
                      "_ZenGroupEmbedding: group_embedding_bag_direct failed"));

      // Passthrough: direct memcpy into the tail of each output row.
      if (num_passthrough_ == 1 && pt_row_features > 0) {
        const Tensor& pt = context->input(pt_input_base);
        const T_output* pt_data = pt.flat<T_output>().data();
        const int64_t pt_bytes = pt_row_features * sizeof(T_output);
        for (int64_t o = 0; o < outer_size; ++o) {
          std::memcpy(out_data + o * output_row_features + gather_row_features,
                      pt_data + o * pt_row_features, pt_bytes);
        }
      }
    }

    ClampBf16Infinities(out_data, output->NumElements());

    zendnnl::error_handling::apilog_info(
        "_ZenGroupEmbedding Compute completed: output_shape=",
        output->shape().DebugString(),
        ", total_elements=", output->NumElements());
  }

 private:
  int num_tables_ = 1;
  int num_lookups_ = 0;
  int embedding_dim_ = -1;
  int gather_axis_ = 0;
  int num_passthrough_ = 0;
  std::vector<int> gathers_per_table_;
};

#define REGISTER_GROUP_EMBEDDING_KERNEL(T_table, T_indices, T_output) \
  REGISTER_KERNEL_BUILDER(Name("_ZenGroupEmbedding")                  \
                              .Device(DEVICE_CPU)                     \
                              .TypeConstraint<T_table>("T_table")     \
                              .TypeConstraint<T_indices>("T_indices") \
                              .TypeConstraint<T_output>("T_output"),  \
                          ZenGroupEmbeddingOp<T_table, T_indices, T_output>);

REGISTER_GROUP_EMBEDDING_KERNEL(float, int64, Eigen::bfloat16);
REGISTER_GROUP_EMBEDDING_KERNEL(float, int32, Eigen::bfloat16);
REGISTER_GROUP_EMBEDDING_KERNEL(float, int64, float);
REGISTER_GROUP_EMBEDDING_KERNEL(float, int32, float);
REGISTER_GROUP_EMBEDDING_KERNEL(Eigen::bfloat16, int64, Eigen::bfloat16);
REGISTER_GROUP_EMBEDDING_KERNEL(Eigen::bfloat16, int32, Eigen::bfloat16);

#undef REGISTER_GROUP_EMBEDDING_KERNEL

}  // namespace amd_cpu_plugin
