/*******************************************************************************
 * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights
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

    OP_REQUIRES_OK(context,
                   context->GetAttr("is_cache_weight", &is_cache_weight_));

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

  using dims = zendnn::memory::dims;

  // This function adjusts the dimensions of an input tensor to match the
  // dimensions of an output tensor.
  void ExpandInputDimsToOutputShape(const TensorShape &input_shape,
                                    const TensorShape &output_shape,
                                    dims *reshaped_dims) {
    auto ndims_input = input_shape.dims();
    auto ndims_output = output_shape.dims();
    auto dim_offset = ndims_output - ndims_input;
    DCHECK(dim_offset > 0);
    reshaped_dims->clear();
    reshaped_dims->resize(ndims_output, 1);
    auto input_dims = input_shape.dim_sizes();
    for (int dim_idx = 0; dim_idx < ndims_input; ++dim_idx) {
      reshaped_dims->at(dim_idx + dim_offset) = input_dims[dim_idx];
    }
  }

  // Extracts dimensions from a TensorFlow shape into a vector of int64_t.
  std::vector<int64_t> ExtractDimsFromTFShape(
      const amd_cpu_plugin::TensorShape &shape) {
    std::vector<int64_t> dims;
    for (int i = 0; i < shape.dims(); ++i) {
      dims.push_back(shape.dim_size(i));
    }
    return dims;
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

    if (!is_float) {
      // Check for the BF16 support on the machine.
      bool result = tensorflow::port::TestCPUFeature(
          tensorflow::port::CPUFeature::AVX512F);
      OP_REQUIRES(
          context, result,
          errors::Internal(
              "BF16 AVX512 instruction set is not supported in the machine."));
    }

    ZenExecutor *ex = ex->getInstance();
    engine eng = ex->getEngine();
    stream s = ex->getStream();
    using tag = memory::format_tag;
    using dt = memory::data_type;
    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    const auto ndims_lhs = lhs.dims();
    const auto ndims_rhs = rhs.dims();
    const int ndims = lhs.dims();
    OP_REQUIRES(
        context, ndims == 3 || ndims == 4,
        errors::InvalidArgument("BatchMatMul is supported for 3D and 4D only"));
    // Kernel only accepts 3D and 4D tensors.
    // Since it's hardcoded as 3D and 4D hence only tested with 3D and 4D
    // tensors.
    Scalar *input_array;
    Scalar *filter_array;
    Scalar *output_array;
    uint64 M, N, K;

    MatMulBCast bcast(lhs.shape().dim_sizes(), rhs.shape().dim_sizes());
    OP_REQUIRES(
        context, bcast.IsValid(),
        errors::InvalidArgument(
            "In[0] and In[1] must have compatible batch dimensions: ",
            lhs.shape().DebugString(), " vs. ", rhs.shape().DebugString()));

    TensorShape out_shape = bcast.output_batch_shape();

    auto lhs_rows = lhs.dim_size(ndims_lhs - 2);
    auto lhs_cols = lhs.dim_size(ndims_lhs - 1);
    auto rhs_rows = rhs.dim_size(ndims_rhs - 2);
    auto rhs_cols = rhs.dim_size(ndims_rhs - 1);

    if (adj_x_) std::swap(lhs_rows, lhs_cols);
    if (adj_y_) std::swap(rhs_rows, rhs_cols);

    out_shape.AddDim(lhs_rows);
    out_shape.AddDim(rhs_cols);
    Tensor *output = nullptr;

    const TensorShape &lhs_shape = lhs.shape();
    const auto ndims_lhs1 = lhs_shape.dims();
    const TensorShape &rhs_shape = rhs.shape();
    const auto ndims_rhs1 = rhs_shape.dims();
    const auto ndims_out = out_shape.dims();

    auto lhs_dims = ExtractDimsFromTFShape(lhs_shape);
    auto rhs_dims = ExtractDimsFromTFShape(rhs_shape);
    auto out_dims = ExtractDimsFromTFShape(out_shape);

    // TODO(plugin) : Added the below flags to differentiate the broadcasting
    // path. Clean it up in future release.
    bool lhs_flag = false, rhs_flag = false;

    // Check if the number of dimensions in lhs is less than the output
    // dimensions.
    if (ndims_lhs1 < ndims_out) {
      ExpandInputDimsToOutputShape(
          lhs_shape, out_shape,
          &lhs_dims);  // Expand lhs dimensions to match
                       // the output shape dimensions.
      lhs_flag = true;
    }

    // Check if the number of dimensions in rhs is less than the output
    // dimensions.
    if (ndims_rhs1 < ndims_out) {
      ExpandInputDimsToOutputShape(
          rhs_shape, out_shape,
          &rhs_dims);  // Expand rhs dimensions to match
                       // the output shape dimensions.
      rhs_flag = true;
    }

    Tensor rhs_reshaped_tensor;
    if (rhs_flag) {
      // TODO(plugin) : Remove the below copy for broadcasting as well.
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<Scalar>::value,
                                  TensorShape(rhs_dims), &rhs_reshaped_tensor));
      std::copy(rhs.flat<Scalar>().data(),
                rhs.flat<Scalar>().data() + rhs.NumElements(),
                rhs_reshaped_tensor.flat<Scalar>().data());
    }

    Tensor lhs_reshaped_tensor;
    if (lhs_flag) {
      // TODO(plugin) : Remove the below copy for broadcasting as well.
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<Scalar>::value,
                                  TensorShape(lhs_dims), &lhs_reshaped_tensor));
      std::copy(lhs.flat<Scalar>().data(),
                lhs.flat<Scalar>().data() + lhs.NumElements(),
                lhs_reshaped_tensor.flat<Scalar>().data());
    }

    if (ndims == 4) {
      input_array = lhs_flag
                        ? const_cast<Scalar *>(
                              lhs_reshaped_tensor.tensor<Scalar, 4>().data())
                        : const_cast<Scalar *>(lhs.tensor<Scalar, 4>().data());
      filter_array = rhs_flag
                         ? const_cast<Scalar *>(
                               rhs_reshaped_tensor.tensor<Scalar, 4>().data())
                         : const_cast<Scalar *>(rhs.tensor<Scalar, 4>().data());
    } else {
      input_array = lhs_flag
                        ? const_cast<Scalar *>(
                              lhs_reshaped_tensor.tensor<Scalar, 3>().data())
                        : const_cast<Scalar *>(lhs.tensor<Scalar, 3>().data());
      filter_array = rhs_flag
                         ? const_cast<Scalar *>(
                               rhs_reshaped_tensor.tensor<Scalar, 3>().data())
                         : const_cast<Scalar *>(rhs.tensor<Scalar, 3>().data());
    }

    M = lhs_flag ? lhs_reshaped_tensor.template flat_inner_dims<Scalar, 3>()
                       .dimension(adj_x_ ? 2 : 1)
                 : lhs.template flat_inner_dims<Scalar, 3>().dimension(
                       adj_x_ ? 2 : 1);
    K = lhs_flag ? lhs_reshaped_tensor.template flat_inner_dims<Scalar, 3>()
                       .dimension(adj_x_ ? 1 : 2)
                 : lhs.template flat_inner_dims<Scalar, 3>().dimension(
                       adj_x_ ? 1 : 2);
    N = rhs_flag ? rhs_reshaped_tensor.template flat_inner_dims<Scalar, 3>()
                       .dimension(adj_y_ ? 1 : 2)
                 : rhs.template flat_inner_dims<Scalar, 3>().dimension(
                       adj_y_ ? 1 : 2);

    ZenTensorType out_type =
        is_float ? ZenTensorType::kFloat : ZenTensorType::kBfloat16;

    ZenMemoryPool<Scalar> *zen_pool_buffer = NULL;
    if (zen_enable_mempool % MEMPOOL_TYPE) {
      unsigned int threadID = GetZenTFthreadId(std::this_thread::get_id());
      zen_pool_buffer = ZenMemoryPool<Scalar>::GetZenMemPool(threadID);
      if (zen_pool_buffer) {
        int status = zen_pool_buffer->AcquireZenPoolTensor(
            context, &output, out_shape, zendnn_params_.out_links,
            zendnn_params_.reset, out_type);
        if (status) {
          zen_enable_mempool = 0;
        }
      } else {
        zen_enable_mempool = 0;
      }
    } else if (zen_enable_mempool) {
      int res = cached_buffer_.NumElements();
      Status state = OkStatus();
      if (res <= 0 || res != out_shape.num_elements()) {
        state = context->allocate_temp(
            is_float ? DataType::DT_FLOAT : DataType::DT_BFLOAT16, out_shape,
            &cached_buffer_);
      }
      if (state != OkStatus()) {
        zen_enable_mempool = 0;
      } else {
        output = &cached_buffer_;
        context->set_output(0, *output);
      }
    }
    if (!zen_enable_mempool) {
      // Output tensor is of the following dimensions:
      // [ in_batch, out_rows, out_cols, out_depth ]
      OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
    }

    if (output->NumElements() == 0) {
      return;
    }

    if (lhs.NumElements() == 0 || rhs.NumElements() == 0) {
      functor::SetZeroFunctor<CPUDevice, Scalar> f;
      f(context->eigen_cpu_device(), output->flat<Scalar>());
      return;
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
      src_dims = {lhs_dims[0], lhs_dims[1], M, K};
      weight_dims = {rhs_dims[0], rhs_dims[1], K, N};
      dst_dims = {lhs_dims[0], lhs_dims[1], M, N};
      weight_tag = adj_y_ ? tag::abdc : tag::abcd;
    } else if (ndims == 3) {
      format_tag = tag::abc;
      src_dims = {lhs_dims[0], M, K};
      weight_dims = {rhs_dims[0], K, N};
      dst_dims = {lhs_dims[0], M, N};
      weight_tag = adj_y_ ? tag::acb : tag::abc;
    }

    memory::data_type dtype = is_float ? dt::f32 : dt::bf16;

    memory::desc src_md = memory::desc({src_dims}, dtype, format_tag);
    memory::desc dst_md = memory::desc({dst_dims}, dtype, format_tag);
    memory::desc matmul_weights_md =
        memory::desc({weight_dims}, dtype, weight_tag, is_cache_weight_);
    memory::desc bias_md = memory::desc();
    zendnn::memory user_weights_memory, src_memory, dst_memory;
    src_memory = memory({{src_dims}, dtype, format_tag}, eng, input_array);
    dst_memory = memory({{dst_dims}, dtype, format_tag}, eng, output_array);

    user_weights_memory =
        memory({{weight_dims}, dtype, weight_tag}, eng, filter_array);

    zendnn::primitive_attr matmul_attr;

    if (fusion_enabled) {
      // Create Mul PostOp.
      const Tensor &mul_tensor = context->input(2);
      Scalar *mul_arr = const_cast<Scalar *>(mul_tensor.flat<Scalar>().data());
      memory::dims mul_dims = {1, 1, 1, 1};
      zendnn::post_ops post_ops;
      post_ops.append_binary(algorithm::binary_mul,
                             memory::desc({mul_dims}, dtype, tag::abcd));

      zendnn::memory postop_memory1, postop_memory2;
      postop_memory1 = memory({{mul_dims}, dtype, tag::abcd}, eng, mul_arr);

      if (fused_computation_ == FusedComputationType::kBinaryMulAdd) {
        // Create Add PostOp.
        const Tensor &add_tensor = context->input(3);
        Scalar *add_arr =
            const_cast<Scalar *>(add_tensor.flat<Scalar>().data());
        memory::dims add_dims = {add_tensor.dim_size(0), add_tensor.dim_size(1),
                                 add_tensor.dim_size(2),
                                 add_tensor.dim_size(3)};
        post_ops.append_binary(algorithm::binary_add,
                               memory::desc({add_dims}, dtype, tag::abcd));
        postop_memory2 = memory({{add_dims}, dtype, tag::abcd}, eng, add_arr);
      }
      matmul_attr.set_post_ops(post_ops);

      zendnn::matmul::desc matmul_pd1 =
          zendnn::matmul::desc(src_md, matmul_weights_md, dst_md);
      zendnn::matmul::primitive_desc matmul_pd =
          zendnn::matmul::primitive_desc(matmul_pd1, matmul_attr, eng);
      net.push_back(zendnn::matmul(matmul_pd));
      if (fused_computation_ == FusedComputationType::kBinaryMul) {
        // BatchMatMul + Mul.
        net_args.push_back(
            {{ZENDNN_ARG_SRC, src_memory},
             {ZENDNN_ARG_WEIGHTS, user_weights_memory},
             {ZENDNN_ARG_DST, dst_memory},
             {ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(0) | ZENDNN_ARG_SRC_1,
              postop_memory1}});
      } else {
        // BatchMatMul + Mul + Add.
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
    zendnnInfo(
        ZENDNN_FWKLOG,
        "ZEN-OP-DEF: _ZenBatchMatMul (TF kernel): Compute Is Successful!");
  }

 private:
  bool adj_x_ = false;
  bool adj_y_ = false;
  bool is_cache_weight_ = false;
  ZendnnParameters zendnn_params_;
  // TF_GUARDED_BY allows the user to specify a particular mutex that should be
  // held when accessing the annotated variable. GUARDED_VAR indicates that
  // a shared variable is guarded by some unspecified mutex, for use in rare
  // cases where a valid mutex expression cannot be specified.
  //
  // Tensor to hold output buffer memory.
  Tensor cached_buffer_ TF_GUARDED_BY(mu_);
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
