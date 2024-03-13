/*******************************************************************************
 * Modifications Copyright (c) 2024 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 ******************************************************************************/

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Standard headers.
#include <vector>

// TensorFlow plug-in headers.
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/fill_functor.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/fused_eigen_output_kernels.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_kernel_common.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_mempool.h"
#include "tensorflow_plugin/src/amd_cpu/util/matmul_bcast.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_requires.h"
#include "tensorflow_plugin/src/amd_cpu/util/register_types.h"
#include "tensorflow_plugin/src/amd_cpu/util/zen_utils.h"

namespace amd_cpu_plugin {

typedef Eigen::ThreadPoolDevice CPUDevice;

// The second parameter v2_bcast is set to true if we are using V2 otherwise we
// set it to false.
template <typename Scalar, bool v2_bcast, bool fusion_enabled>
class ZenBatchMatMulOp : public OpKernel {
 public:
  virtual ~ZenBatchMatMulOp() {}

  explicit ZenBatchMatMulOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, InitZendnnParameters(context, &zendnn_params_));

    OP_REQUIRES_OK(context, context->GetAttr("adj_x", &adj_x_));
    OP_REQUIRES_OK(context, context->GetAttr("adj_y", &adj_y_));

    std::vector<FusedComputationPattern> patterns;
    if (fusion_enabled) {
      using FCT = FusedComputationType;

      patterns = {{FCT::kBinaryMul, {"BinaryMul"}},
                  {FCT::kBinaryMulAdd, {"BinaryMul", "Add"}}};

      OP_REQUIRES_OK(context,
                     InitializeFusedComputation(context, "_ZenBatchMatMul",
                                                patterns, &fused_computation_,
                                                &fused_computation_args_));
    }
  }

  void Compute(OpKernelContext *context) override {
    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-DEF: _ZenBatchMatMul (TF kernel): In Compute!");

    const Tensor &lhs = context->input(0);
    const Tensor &rhs = context->input(1);
    bool is_float = std::is_same<Scalar, float>::value;

    zendnnEnv zen_env_obj = readEnv();
    int zen_enable_mempool =
        zendnn_params_.is_eager ? 0 : zen_env_obj.zenEnableMemPool;

    if (is_float) {
      if (!v2_bcast) {
        // Using V1, so check to make sure lhs and rhs dimensions are correct
        // and no broadcasting is needed.
        OP_REQUIRES(context, lhs.dims() == rhs.dims(),
                    errors::InvalidArgument("lhs and rhs has different ndims: ",
                                            lhs.shape().DebugString(), " vs. ",
                                            rhs.shape().DebugString()));
        const int ndims = lhs.dims();
        OP_REQUIRES(
            context, ndims >= 2,
            errors::InvalidArgument("lhs and rhs ndims must be >= 2: ", ndims));
        for (int i = 0; i < ndims - 2; ++i) {
          OP_REQUIRES(context, lhs.dim_size(i) == rhs.dim_size(i),
                      errors::InvalidArgument(
                          "lhs.dim(", i, ") and rhs.dim(", i,
                          ") must be the same: ", lhs.shape().DebugString(),
                          " vs ", rhs.shape().DebugString()));
        }
      } else {
        OP_REQUIRES(
            context, lhs.dims() >= 2,
            errors::InvalidArgument("In[0] ndims must be >= 2: ", lhs.dims()));
        OP_REQUIRES(
            context, rhs.dims() >= 2,
            errors::InvalidArgument("In[1] ndims must be >= 2: ", rhs.dims()));
      }

      // lhs and rhs can have different dimensions.
      const int ndims_lhs = lhs.dims();
      const int ndims_rhs = rhs.dims();

      // Get broadcast information.
      MatMulBCast bcast(lhs.shape().dim_sizes(), rhs.shape().dim_sizes());
      OP_REQUIRES(
          context, bcast.IsValid(),
          errors::InvalidArgument(
              "In[0] and In[1] must have compatible batch dimensions: ",
              lhs.shape().DebugString(), " vs. ", rhs.shape().DebugString()));

      TensorShape out_shape = bcast.output_batch_shape();
      auto batch_size = bcast.output_batch_size();

      auto lhs_rows = lhs.dim_size(ndims_lhs - 2);
      auto lhs_cols = lhs.dim_size(ndims_lhs - 1);
      auto rhs_rows = rhs.dim_size(ndims_rhs - 2);
      auto rhs_cols = rhs.dim_size(ndims_rhs - 1);

      // TF-Vanilla path will override lhs_reshaped and rhs_reshaped.
      auto rhs_reshaped = rhs.template flat_inner_dims<float, 3>();
      auto lhs_reshaped = lhs.template flat_inner_dims<float, 3>();

      const uint64 M = lhs_reshaped.dimension(adj_x_ ? 2 : 1);
      const uint64 K = lhs_reshaped.dimension(adj_x_ ? 1 : 2);
      const uint64 N = rhs_reshaped.dimension(adj_y_ ? 1 : 2);

      // TODO(plugin): Add Switch for TF-vanilla and TF-zendnn when M and N <=
      // 64 TF-Vanilla path(Eigen implementation) is optimal for batched GEMM
      // execution ZenDNN kernel works well beyond above M and N values
      // (Currently flow is going for ZenDNN implementation only).
      if (adj_x_) {
        std::swap(lhs_rows, lhs_cols);
      }
      if (adj_y_) {
        std::swap(rhs_rows, rhs_cols);
      }
      OP_REQUIRES(context, lhs_cols == rhs_rows,
                  errors::InvalidArgument(
                      "lhs mismatch rhs shape: ", lhs_cols, " vs. ", rhs_rows,
                      ": ", lhs.shape().DebugString(), " ",
                      rhs.shape().DebugString(), " ", adj_x_, " ", adj_y_));

      out_shape.AddDim(lhs_rows);
      out_shape.AddDim(rhs_cols);

      // Update the output type.
      ZenTensorType out_type = ZenTensorType::kFloat;
      Tensor *out = nullptr;
      ZenMemoryPool<Scalar> *zen_pool_buffer = NULL;

      // ZenMemPool optimization reuse o/p tensors from the pool. By default its
      // enabled, export ZENDNN_ENABLE_MEMPOOL=0 will disable memory pool
      // optimization.
      // Cases where tensors in pool are not free or requested size is more than
      // available tensor size in Pool, control will fall back to default way of
      // allocation i.e. with allocate_output(..).
      if (zen_enable_mempool % MEMPOOL_TYPE) {
        unsigned int thread_id = GetZenTFthreadId(std::this_thread::get_id());
        zen_pool_buffer = ZenMemoryPool<Scalar>::GetZenMemPool(thread_id);
        if (zen_pool_buffer) {
          int status = zen_pool_buffer->AcquireZenPoolTensor(
              context, &out, out_shape, zendnn_params_.out_links,
              zendnn_params_.reset, out_type);
          if (status) {
            zen_enable_mempool = 0;
          }
        } else {
          zen_enable_mempool = 0;
        }
      } else if (zen_enable_mempool) {
        // Caching the output buffer and reusing it with persistent tensor.
        int res = cached_data_.NumElements();
        if (res <= 0 || res != out_shape.num_elements()) {
          context->allocate_temp(DataType::DT_FLOAT, out_shape, &cached_data_);
        }
        out = &cached_data_;
        context->set_output(0, *out);
      }
      if (!zen_enable_mempool) {
        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
      }

      if (out->NumElements() == 0) {
        return;
      }

      if (lhs.NumElements() == 0 || rhs.NumElements() == 0) {
        functor::SetZeroFunctor<CPUDevice, Scalar> f;
        f(context->eigen_cpu_device(), out->flat<Scalar>());
        return;
      }

      auto out_reshaped = out->template flat_inner_dims<float, 3>();
      std::vector<int> m_array(batch_size, M);
      std::vector<int> n_array(batch_size, N);
      std::vector<int> k_array(batch_size, K);
      std::vector<int> lda_array(batch_size, adj_x_ ? M : K);
      std::vector<int> ldb_array(batch_size, adj_y_ ? K : N);
      std::vector<int> ldc_array(batch_size, N);
      std::vector<float> alpha_array(batch_size, 1.0);
      std::vector<float> beta_array(batch_size, 0.0);
      std::vector<int> group_size(1, batch_size);
      std::vector<const float *> a_array;
      std::vector<const float *> b_array;
      std::vector<float *> c_array;
      std::vector<int> add_shape;
      std::vector<const float *> add_array(out->dim_size(0), nullptr);
      a_array.reserve(batch_size);
      b_array.reserve(batch_size);
      c_array.reserve(batch_size);

      if (!bcast.IsBroadcastingRequired()) {
        for (int64 i = 0; i < batch_size; i++) {
          a_array.push_back(&lhs_reshaped(i, 0, 0));
          b_array.push_back(&rhs_reshaped(i, 0, 0));
          c_array.push_back(&out_reshaped(i, 0, 0));
        }
      } else {
        // Broadcasting is needed, so get the mapping from flattened output
        // batch indices to x's and y's flattened batch indices.
        const std::vector<int64> &a_batch_indices = bcast.x_batch_indices();
        const std::vector<int64> &b_batch_indices = bcast.y_batch_indices();

        for (int64 i = 0; i < batch_size; i++) {
          a_array.push_back(&lhs_reshaped(a_batch_indices[i], 0, 0));
          b_array.push_back(&rhs_reshaped(b_batch_indices[i], 0, 0));
          c_array.push_back(&out_reshaped(i, 0, 0));
        }
      }

      bool cblasRowMajor = 1;
      float mul_node = 1.0f;

      if (fusion_enabled) {
        // BatchMatMul + Mul
        const Tensor &mul_tensor = context->input(2);
        mul_node = mul_tensor.flat<float>().data()[0];
        if (fused_computation_ == FusedComputationType::kBinaryMulAdd) {
          // BatchmatMul + Mul + Add
          const Tensor &add_tensor = context->input(3);
          auto add_reshaped = add_tensor.template flat_inner_dims<float, 3>();
          add_shape = {add_reshaped.dimension(0), add_reshaped.dimension(1),
                       add_reshaped.dimension(2)};
          for (int64 i = 0; i < out->dim_size(0); i++) {
            add_array[i] = &add_reshaped(i, 0, 0);
          }
        }
      }

      zenBatchMatMul(cblasRowMajor, adj_x_, adj_y_, &m_array[0], &n_array[0],
                     &k_array[0], &alpha_array[0], &a_array[0], &lda_array[0],
                     &b_array[0], &ldb_array[0], &beta_array[0], &c_array[0],
                     &ldc_array[0], 1, &group_size[0], &add_array[0],
                     &add_shape[0], mul_node, out->dim_size(0));

      // If ZenMemPool Optimization is enabled(default), update the state of
      // memory pool based on input_array address.
      if ((zen_env_obj.zenEnableMemPool % MEMPOOL_TYPE) &&
          !zendnn_params_.is_eager && zen_pool_buffer) {
        zen_pool_buffer->ZenMemPoolFree(
            context, static_cast<void *>(const_cast<float *>(a_array[0])));
        zen_pool_buffer->ZenMemPoolFree(
            context, static_cast<void *>(const_cast<float *>(b_array[0])));
      }
    } else {
      // Check for the BF16 support on the machine.
      bool result = tensorflow::port::TestCPUFeature(
          tensorflow::port::CPUFeature::AVX512F);
      OP_REQUIRES(
          context, result,
          errors::Internal(
              "BF16 AVX512 instruction set is not supported in the machine."));
      // BF16 BatchMatMul execution.
      ZenExecutor *ex = ex->getInstance();
      engine eng = ex->getEngine();
      stream s = ex->getStream();
      using tag = memory::format_tag;
      using dt = memory::data_type;
      std::vector<primitive> net;
      std::vector<std::unordered_map<int, memory>> net_args;
      const int ndims = lhs.dims();
      OP_REQUIRES(context, ndims == 3 || ndims == 4,
                  errors::InvalidArgument(
                      "BatchMatMul is supported for 3D and 4D only for BF16"));
      // BF16 kernel only accepts 3D and 4D tensors.
      // since it's hardcoded as 3D and 4D hence only tested with 3D and 4D
      // tensors.
      Scalar *input_array;
      Scalar *filter_array;
      Scalar *output_array;
      uint64 M, N, K;
      if (ndims == 4) {
        input_array = const_cast<Scalar *>(lhs.tensor<Scalar, 4>().data());
        filter_array = const_cast<Scalar *>(rhs.tensor<Scalar, 4>().data());
        auto rhs_reshaped = rhs.template flat_inner_dims<Scalar, 3>();
        auto lhs_reshaped = lhs.template flat_inner_dims<Scalar, 3>();
        M = lhs_reshaped.dimension(adj_x_ ? 2 : 1);
        K = lhs_reshaped.dimension(adj_x_ ? 1 : 2);
        N = rhs_reshaped.dimension(adj_y_ ? 1 : 2);
      } else {
        input_array = const_cast<Scalar *>(lhs.tensor<Scalar, 3>().data());
        filter_array = const_cast<Scalar *>(rhs.tensor<Scalar, 3>().data());
        M = adj_x_ ? lhs.dim_size(2) : lhs.dim_size(1);
        K = adj_x_ ? lhs.dim_size(1) : lhs.dim_size(2);
        N = adj_y_ ? rhs.dim_size(1) : rhs.dim_size(2);
      }
      MatMulBCast bcast(lhs.shape().dim_sizes(), rhs.shape().dim_sizes());
      OP_REQUIRES(
          context, bcast.IsValid(),
          errors::InvalidArgument(
              "In[0] and In[1] must have compatible batch dimensions: ",
              lhs.shape().DebugString(), " vs. ", rhs.shape().DebugString()));

      TensorShape out_shape = bcast.output_batch_shape();
      out_shape.AddDim(M);
      out_shape.AddDim(N);
      ZenTensorType out_type = ZenTensorType::kBfloat16;
      Tensor *output = nullptr;

      ZenMemoryPool<Scalar> *zen_pool_buffer = NULL;
      if (zen_enable_mempool % MEMPOOL_TYPE) {
        unsigned int threadID = GetZenTFthreadId(std::this_thread::get_id());
        zen_pool_buffer = ZenMemoryPool<Scalar>::GetZenMemPool(threadID);
        if (zen_pool_buffer) {
          int status = zen_pool_buffer->AcquireZenPoolTensor(
              context, &output, out_shape, zendnn_params_.out_links,
              zendnn_params_.reset, out_type);
          if (status) {
            zen_enable_mempool = false;
          }
        } else {
          zen_enable_mempool = false;
        }
      } else if (zen_enable_mempool) {
        int res = cached_data_.NumElements();
        if (res <= 0 || res != out_shape.num_elements()) {
          context->allocate_temp(DataType::DT_BFLOAT16, out_shape,
                                 &cached_data_);
        }
        output = &cached_data_;
        context->set_output(0, *output);
      }
      if (!zen_enable_mempool) {
        // Output tensor is of the following dimensions:
        // [ in_batch, out_rows, out_cols, out_depth ]
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, out_shape, &output));
      }

      if (ndims == 4) {
        auto output_map = output->tensor<Scalar, 4>();
        output_array = const_cast<Scalar *>(output_map.data());
      } else {
        auto output_map = output->tensor<Scalar, 3>();
        output_array = const_cast<Scalar *>(output_map.data());
      }
      tag format_tag;
      memory::dims src_dims;
      memory::dims weight_dims;
      memory::dims dst_dims;
      tag weight_tag;
      memory::dims bias_dims = {1, 1, 1, N};
      if (ndims == 4) {
        format_tag = tag::abcd;
        src_dims = {lhs.dim_size(0), lhs.dim_size(1), M, K};
        weight_dims = {rhs.dim_size(0), rhs.dim_size(1), K, N};
        dst_dims = {lhs.dim_size(0), lhs.dim_size(1), M, N};
        weight_tag = adj_y_ ? tag::abdc : tag::abcd;
      } else if (ndims == 3) {
        format_tag = tag::abc;
        src_dims = {lhs.dim_size(0), M, K};
        weight_dims = {rhs.dim_size(0), K, N};
        dst_dims = {lhs.dim_size(0), M, N};
        weight_tag = adj_y_ ? tag::acb : tag::abc;
      }

      memory::desc src_md = memory::desc({src_dims}, dt::bf16, format_tag);
      memory::desc dst_md = memory::desc({dst_dims}, dt::bf16, format_tag);
      memory::desc matmul_weights_md =
          memory::desc({weight_dims}, dt::bf16, weight_tag);
      memory::desc bias_md = memory::desc();
      zendnn::memory user_weights_memory, src_memory, dst_memory;
      src_memory = memory({{src_dims}, dt::bf16, format_tag}, eng, input_array);
      dst_memory =
          memory({{dst_dims}, dt::bf16, format_tag}, eng, output_array);

      user_weights_memory =
          memory({{weight_dims}, dt::bf16, weight_tag}, eng, filter_array);

      zendnn::primitive_attr matmul_attr;

      if (fusion_enabled) {
        // Mul PostOp for BatchMatMul
        const Tensor &mul_tensor = context->input(2);
        Scalar *mul_arr =
            const_cast<Scalar *>(mul_tensor.flat<Scalar>().data());
        memory::dims mul_dims = {1, 1, 1, 1};
        zendnn::post_ops post_ops;
        post_ops.append_binary(algorithm::binary_mul,
                               memory::desc({mul_dims}, dt::bf16, tag::abcd));

        zendnn::memory postop_memory1, postop_memory2;
        postop_memory1 =
            memory({{mul_dims}, dt::bf16, tag::abcd}, eng, mul_arr);

        if (fused_computation_ == FusedComputationType::kBinaryMulAdd) {
          // Add PostOp for BatchMatMul
          const Tensor &add_tensor = context->input(3);
          Scalar *add_arr =
              const_cast<Scalar *>(add_tensor.flat<Scalar>().data());
          memory::dims add_dims = {
              add_tensor.dim_size(0), add_tensor.dim_size(1),
              add_tensor.dim_size(2), add_tensor.dim_size(3)};
          post_ops.append_binary(algorithm::binary_add,
                                 memory::desc({add_dims}, dt::bf16, tag::abcd));
          postop_memory2 =
              memory({{add_dims}, dt::bf16, tag::abcd}, eng, add_arr);
        }
        matmul_attr.set_post_ops(post_ops);

        zendnn::matmul::desc matmul_pd1 =
            zendnn::matmul::desc(src_md, matmul_weights_md, dst_md);
        zendnn::matmul::primitive_desc matmul_pd =
            zendnn::matmul::primitive_desc(matmul_pd1, matmul_attr, eng);
        net.push_back(zendnn::matmul(matmul_pd));
        if (fused_computation_ == FusedComputationType::kBinaryMul) {
          // BatchMatMul + Mul
          net_args.push_back(
              {{ZENDNN_ARG_SRC, src_memory},
               {ZENDNN_ARG_WEIGHTS, user_weights_memory},
               {ZENDNN_ARG_DST, dst_memory},
               {ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(0) | ZENDNN_ARG_SRC_1,
                postop_memory1}});
        } else {
          // BatchMatMul + Mul + Add
          net_args.push_back(
              {{ZENDNN_ARG_SRC, src_memory},
               {ZENDNN_ARG_WEIGHTS, user_weights_memory},
               {ZENDNN_ARG_DST, dst_memory},
               {ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(0) | ZENDNN_ARG_SRC_1,
                postop_memory1},
               {ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(1) | ZENDNN_ARG_SRC_1,
                postop_memory2}});
        }
      } else {
        zendnn::matmul::desc matmul_pd1 =
            zendnn::matmul::desc(src_md, matmul_weights_md, bias_md, dst_md);
        zendnn::matmul::primitive_desc matmul_pd =
            zendnn::matmul::primitive_desc(matmul_pd1, matmul_attr, eng);

        net.push_back(zendnn::matmul(matmul_pd));
        net_args.push_back({{ZENDNN_ARG_SRC, src_memory},
                            {ZENDNN_ARG_WEIGHTS, user_weights_memory},
                            {ZENDNN_ARG_DST, dst_memory}});
      }
      assert(net.size() == net_args.size() && "something is missing");
      for (size_t i = 0; i < net.size(); ++i) {
        net.at(i).execute(s, net_args.at(i));
      }

      if ((zen_env_obj.zenEnableMemPool % MEMPOOL_TYPE) && zen_pool_buffer) {
        zen_pool_buffer->ZenMemPoolFree(context,
                                        static_cast<void *>(input_array));
        zen_pool_buffer->ZenMemPoolFree(context,
                                        static_cast<void *>(filter_array));
      }
    }
    zendnnInfo(
        ZENDNN_FWKLOG,
        "ZEN-OP-DEF: _ZenBatchMatMul (TF kernel): Compute Is Successful!");
  }

 private:
  bool adj_x_, adj_y_;
  ZendnnParameters zendnn_params_;
  // TF_GUARDED_BY allows the user to specify a particular mutex that should be
  // held when accessing the annotated variable. GUARDED_VAR indicates that
  // a shared variable is guarded by some unspecified mutex, for use in rare
  // cases where a valid mutex expression cannot be specified.
  //
  // Tensor to hold output buffer memory.
  Tensor cached_data_ TF_GUARDED_BY(mu_);
  FusedComputationType fused_computation_ = FusedComputationType::kUndefined;
  FusedComputationArgs fused_computation_args_;
};
#define REGISTER_BATCH_MATMUL_KERNELS(TYPE)                                   \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_ZenBatchMatMul").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"),   \
      ZenBatchMatMulOp<TYPE, false, false>);                                  \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_ZenBatchMatMulV2").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      ZenBatchMatMulOp<TYPE, true, false>);                                   \
  REGISTER_KERNEL_BUILDER(Name("_ZenFusedBatchMatMulV2")                      \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<TYPE>("T"),                     \
                          ZenBatchMatMulOp<TYPE, true, true>);

TF_CALL_float(REGISTER_BATCH_MATMUL_KERNELS);
TF_CALL_bfloat16(REGISTER_BATCH_MATMUL_KERNELS);
#undef REGISTER_BATCH_MATMUL_KERNELS

}  // namespace amd_cpu_plugin
