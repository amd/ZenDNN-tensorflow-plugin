/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
 ******************************************************************************/

// TensorFlow plug-in headers.
#include <cmath>
#include <cstring>
#include <vector>

#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_kernel_common.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_zendnnl_utils.h"
#include "tensorflow_plugin/src/amd_cpu/util/errors.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_kernel.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_requires.h"
#include "tensorflow_plugin/src/amd_cpu/util/register_types.h"
#include "tensorflow_plugin/src/amd_cpu/util/tensor_format.h"
#include "tensorflow_plugin/src/amd_cpu/util/zen_utils.h"

namespace amd_cpu_plugin {

namespace {

enum class Combiner { kSum, kMean, kSqrtN };

Combiner ParseCombiner(const string& s) {
  if (s == "sum") return Combiner::kSum;
  if (s == "mean") return Combiner::kMean;
  return Combiner::kSqrtN;
}

}  // namespace

// ============================================================================
// _ZenSafeEmbeddingLookupSparse kernel
//
// Fuses key parts of the safe_embedding_lookup_sparse subgraph:
//   1. Embedding lookup via GatherV2 — lookup rows from params
//   2. SparseSegment{Sum,Mean,SqrtN} — reduce gathered rows per segment
//   3. Zero empty rows — mask originally-empty rows to zeros
// ============================================================================

template <typename T, typename Tindices>
class ZenFusedEmbeddingLookupSparseOp : public OpKernel {
 public:
  ~ZenFusedEmbeddingLookupSparseOp() {}

  explicit ZenFusedEmbeddingLookupSparseOp(OpKernelConstruction* context)
      : OpKernel(context) {
    string combiner_str;
    OP_REQUIRES_OK(context, context->GetAttr("combiner", &combiner_str));
    combiner_ = ParseCombiner(combiner_str);
    OP_REQUIRES_OK(context,
                   context->GetAttr("has_adjust_shape", &has_adjust_shape_));
  }

  void Compute(OpKernelContext* context) override {
    // Input 0: params [num_embeddings, embedding_dim]
    const Tensor& params = context->input(0);
    // Input 1: sp_indices [nnz, rank] (int64)
    const Tensor& sp_indices = context->input(1);
    // Input 2: sp_values [nnz] (Tindices — indices into params)
    const Tensor& sp_values = context->input(2);
    // Input 3: sp_dense_shape [rank] (int64)
    const Tensor& sp_dense_shape = context->input(3);
    // Input 4: default_value (scalar Tindices)
    const Tensor& default_value = context->input(4);
    // Input 5: orig_dense_shape [orig_rank] (int64)
    const Tensor& orig_dense_shape = context->input(5);

    OP_REQUIRES(context, params.dims() == 2,
                errors::InvalidArgument(
                    "_ZenSafeEmbeddingLookupSparse: params must be rank-2, "
                    "got rank ",
                    params.dims()));
    OP_REQUIRES(context, sp_indices.dims() == 2,
                errors::InvalidArgument(
                    "_ZenSafeEmbeddingLookupSparse: sp_indices must be "
                    "rank-2, got rank ",
                    sp_indices.dims()));
    OP_REQUIRES(context, sp_values.dims() == 1,
                errors::InvalidArgument(
                    "_ZenSafeEmbeddingLookupSparse: sp_values must be "
                    "rank-1, got rank ",
                    sp_values.dims()));

    const int64 num_embeddings = params.dim_size(0);
    const int64 embedding_dim = params.dim_size(1);
    const int64 nnz = sp_values.dim_size(0);

    // Dense shape gives us the batch size (first dim).
    OP_REQUIRES(context, sp_dense_shape.NumElements() >= 1,
                errors::InvalidArgument(
                    "_ZenSafeEmbeddingLookupSparse: sp_dense_shape must have "
                    "at least 1 element"));
    const int64* dense_shape_data = sp_dense_shape.flat<int64>().data();
    const int64 batch_size = dense_shape_data[0];

    // Build output shape:
    // If has_adjust_shape: use orig_dense_shape batch dims + embedding_dim.
    // Otherwise: [batch_size, embedding_dim].
    TensorShape out_shape;
    if (has_adjust_shape_) {
      const int64* orig_ds = orig_dense_shape.flat<int64>().data();
      const int orig_rank = orig_dense_shape.NumElements();
      // Copy all dims except the last from orig_dense_shape, append embed_dim.
      for (int d = 0; d < orig_rank - 1; ++d) {
        out_shape.AddDim(orig_ds[d]);
      }
      out_shape.AddDim(embedding_dim);
    } else {
      out_shape.AddDim(batch_size);
      out_shape.AddDim(embedding_dim);
    }
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    if (batch_size == 0 || embedding_dim == 0) return;

    // Zero the output — empty rows stay zero, and we accumulate into it.
    memset(output->data(), 0, output->NumElements() * sizeof(T));

    if (nnz == 0) return;

    const int64* indices_data = sp_indices.flat<int64>().data();
    const Tindices* values_data = sp_values.flat<Tindices>().data();
    const T* params_data = params.flat<T>().data();
    T* output_data = output->flat<T>().data();

    // Get the default fill value for SparseFillEmptyRows.
    Tindices default_val = default_value.scalar<Tindices>()();

    // ---------------------------------------------------------------
    // Fused: GreaterEqual + Where + GatherV2 + SparseFillEmptyRows +
    //        SparseSegmentReduce in a single pass.
    //
    // The upstream filter chain (GreaterEqual >= 0, Where, GatherV2) is
    // absorbed: we skip entries where sp_values[i] < 0 (invalid hashed
    // entries). For each valid entry, accumulate the embedding into the
    // corresponding segment row.
    // ---------------------------------------------------------------

    const int64 indices_cols = sp_indices.dim_size(1);

    // Count valid entries per segment for combiner normalization.
    std::vector<int64> segment_counts(batch_size, 0);

    // Accumulate embeddings per segment, filtering negatives inline.
    for (int64 i = 0; i < nnz; ++i) {
      const Tindices val = values_data[i];

      // Filter: skip entries with negative values (absorbs GreaterEqual >= 0).
      if (val < static_cast<Tindices>(0)) continue;

      const int64 segment = indices_data[i * indices_cols];
      if (segment < 0 || segment >= batch_size) continue;

      if (val >= static_cast<Tindices>(num_embeddings)) continue;

      // Accumulate: output[segment] += params[val]
      const T* src = params_data + static_cast<int64>(val) * embedding_dim;
      T* dst = output_data + segment * embedding_dim;
      for (int64 d = 0; d < embedding_dim; ++d) {
        dst[d] += src[d];
      }
      segment_counts[segment]++;
    }

    // ---------------------------------------------------------------
    // Step 3: Apply combiner normalization.
    // Empty rows (count == 0) stay zero. This fusion only matches the
    // SelectV2 branch of safe_embedding_lookup_sparse, which corresponds
    // to default_id=None in TF. TF fills empty rows with params[0] via
    // SparseFillEmptyRows then zeros them out with SelectV2 — so the
    // net result for empty rows is always a zero vector.
    // ---------------------------------------------------------------
    if (combiner_ != Combiner::kSum) {
      for (int64 seg = 0; seg < batch_size; ++seg) {
        const int64 count = segment_counts[seg];
        if (count == 0) continue;

        T* dst = output_data + seg * embedding_dim;
        if (combiner_ == Combiner::kMean) {
          const T inv = static_cast<T>(1.0) / static_cast<T>(count);
          for (int64 d = 0; d < embedding_dim; ++d) {
            dst[d] *= inv;
          }
        } else if (combiner_ == Combiner::kSqrtN) {
          const T inv = static_cast<T>(1.0) /
                        static_cast<T>(std::sqrt(static_cast<double>(count)));
          for (int64 d = 0; d < embedding_dim; ++d) {
            dst[d] *= inv;
          }
        }
      }
    }
  }

 private:
  Combiner combiner_ = Combiner::kMean;
  bool has_adjust_shape_ = false;
};

// Register kernels for all combinations of T and Tindices.
#define REGISTER_EMBEDDING_LOOKUP_SPARSE_KERNEL(T, Tindices)         \
  REGISTER_KERNEL_BUILDER(Name("_ZenSafeEmbeddingLookupSparse")      \
                              .Device(DEVICE_CPU)                    \
                              .TypeConstraint<T>("T")                \
                              .TypeConstraint<Tindices>("Tindices"), \
                          ZenFusedEmbeddingLookupSparseOp<T, Tindices>);

REGISTER_EMBEDDING_LOOKUP_SPARSE_KERNEL(float, int32);
REGISTER_EMBEDDING_LOOKUP_SPARSE_KERNEL(float, int64);
REGISTER_EMBEDDING_LOOKUP_SPARSE_KERNEL(Eigen::bfloat16, int32);
REGISTER_EMBEDDING_LOOKUP_SPARSE_KERNEL(Eigen::bfloat16, int64);
#undef REGISTER_EMBEDDING_LOOKUP_SPARSE_KERNEL

}  // namespace amd_cpu_plugin
