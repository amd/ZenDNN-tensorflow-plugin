/*******************************************************************************
 * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 ******************************************************************************/

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <limits>
#include <memory>
#include <vector>

#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/fill_functor.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/fused_eigen_output_kernels.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_kernel_common.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_matmul_kernel_util.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_mempool.h"
#include "tensorflow_plugin/src/amd_cpu/util/errors.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_kernel.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_requires.h"
#include "tensorflow_plugin/src/amd_cpu/util/padding.h"
#include "tensorflow_plugin/src/amd_cpu/util/register_types.h"
#include "tensorflow_plugin/src/amd_cpu/util/tensor_format.h"

namespace amd_cpu_plugin {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename T>
struct LaunchZenFusedMatMulOp {
  void operator()(OpKernelContext *context, const Tensor &a, const Tensor &b,
                  ZenMatMulParams matmul_params, FusedComputationType fusion,
                  const FusedComputationArgs &fusion_args, Tensor *output,
                  bool is_biasadd) {
    T *bias_ptr = nullptr;
    if (is_biasadd) {
      const Tensor &bias = context->input(2);
      if (BiasAddArgs<T>::IsSupported(fusion)) {
        for (int i = 0; i < bias.dims() - 1; i++) {
          OP_REQUIRES(context, bias.dim_size(i) == 1,
                      errors::InvalidArgument(
                          "For bias_dims > 1, all except the "
                          "last dimension (channel) must be 1, got: ",
                          bias.shape().DebugString()));
        }
      }
      bias_ptr = const_cast<T *>(bias.flat<T>().data());
    }

    matmul_params.is_biasadd = is_biasadd;

    auto a_ptr = const_cast<T *>(a.template flat<T>().data());
    auto b_ptr = const_cast<T *>(b.template flat<T>().data());
    auto c_ptr = (output->template flat<T>().data());

    switch (fusion) {
      case FusedComputationType::kBiasAdd: {
        ZenMatMulPrimitive<T, T, T, T> *matmul_prim =
            ZenMatMulPrimitiveFactory<T, T, T, T>::Get(matmul_params, 0);
        matmul_prim->Execute(a_ptr, b_ptr, bias_ptr, c_ptr, is_biasadd);
        break;
      }
      case FusedComputationType::kBiasAddWithAdd: {
        matmul_params.post_op_params.push_back({"sum", {1.0}});
        ZenMatMulPrimitive<T, T, T, T> *matmul_prim =
            ZenMatMulPrimitiveFactory<T, T, T, T>::Get(matmul_params, 1);
        matmul_prim->Execute(a_ptr, b_ptr, bias_ptr, c_ptr, is_biasadd);
        break;
      }
      case FusedComputationType::kBiasAddWithRelu: {
        matmul_params.post_op_params.push_back({"relu", {1.0, 0.0, 0.0}});
        ZenMatMulPrimitive<T, T, T, T> *matmul_prim =
            ZenMatMulPrimitiveFactory<T, T, T, T>::Get(matmul_params, 0);
        matmul_prim->Execute(a_ptr, b_ptr, bias_ptr, c_ptr, is_biasadd);
        break;
      }
      case FusedComputationType::kBiasAddWithSigmoid: {
        matmul_params.post_op_params.push_back({"sigmoid", {1.0, 1.0, 0.0}});
        ZenMatMulPrimitive<T, T, T, T> *matmul_prim =
            ZenMatMulPrimitiveFactory<T, T, T, T>::Get(matmul_params, 0);
        matmul_prim->Execute(a_ptr, b_ptr, bias_ptr, c_ptr, is_biasadd);
        break;
      }
      case FusedComputationType::kBiasAddWithAddAndRelu: {
        matmul_params.post_op_params.push_back({"sum", {1.0}});
        matmul_params.post_op_params.push_back({"relu", {1.0, 0.0, 0.0}});
        ZenMatMulPrimitive<T, T, T, T> *matmul_prim =
            ZenMatMulPrimitiveFactory<T, T, T, T>::Get(matmul_params, 1);
        matmul_prim->Execute(a_ptr, b_ptr, bias_ptr, c_ptr, is_biasadd);
        break;
      }
      case FusedComputationType::kBiasAddWithGeluApproximate: {
        matmul_params.post_op_params.push_back(
            {"GeluApproximate", {1.0, 1.0, 0.0}});
        ZenMatMulPrimitive<T, T, T, T> *matmul_prim =
            ZenMatMulPrimitiveFactory<T, T, T, T>::Get(matmul_params, 1);
        matmul_prim->Execute(a_ptr, b_ptr, bias_ptr, c_ptr, is_biasadd);
        break;
      }
      case FusedComputationType::kBiasAddWithGeluExact: {
        matmul_params.post_op_params.push_back({"GeluExact", {1.0, 1.0, 0.0}});
        ZenMatMulPrimitive<T, T, T, T> *matmul_prim =
            ZenMatMulPrimitiveFactory<T, T, T, T>::Get(matmul_params, 1);
        matmul_prim->Execute(a_ptr, b_ptr, bias_ptr, c_ptr, is_biasadd);
        break;
      }
      case FusedComputationType::kRelu: {
        matmul_params.post_op_params.push_back({"relu", {1.0, 0.0, 0.0}});
        ZenMatMulPrimitive<T, T, T, T> *matmul_prim =
            ZenMatMulPrimitiveFactory<T, T, T, T>::Get(matmul_params, 1);
        matmul_prim->Execute(a_ptr, b_ptr, bias_ptr, c_ptr, false);
        break;
      }
      case FusedComputationType::kBiasAddWithRelu6:
        OP_REQUIRES_OK(context, errors::Internal("Fusion type not supported"));
        break;
      case FusedComputationType::kBiasAddWithElu:
        OP_REQUIRES_OK(context, errors::Internal("Fusion type not supported"));
        break;
      case FusedComputationType::kUndefined:
        OP_REQUIRES_OK(context, errors::Internal("Fusion type is undefined"));
        break;
      default:
        OP_REQUIRES_OK(context,
                       errors::Internal("Fusion type is not supported"));
    }
  }
};

template <typename Device, typename T, bool is_bias_add_gelu = false,
          bool is_fused = false>
class ZenMatMulOp : public OpKernel {
 public:
  explicit ZenMatMulOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, InitZendnnParameters(context, &zendnn_params_));

    OP_REQUIRES_OK(context, context->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(context, context->GetAttr("transpose_b", &transpose_b_));

    std::vector<FusedComputationPattern> patterns;
    is_reshape_ = false;
    if (is_fused) {
      OP_REQUIRES_OK(context, context->GetAttr("is_reshape", &is_reshape_));
      using FCT = FusedComputationType;
      patterns = {
          {FCT::kBiasAdd, {"BiasAdd"}},
          {FCT::kBiasAddWithAdd, {"BiasAdd", "Add"}},
          {FCT::kBiasAddWithRelu, {"BiasAdd", "Relu"}},
          {FCT::kBiasAddWithSigmoid, {"BiasAdd", "Sigmoid"}},
          {FCT::kBiasAddWithGeluExact, {"BiasAdd", "GeluExact"}},
          {FCT::kBiasAddWithAddAndRelu, {"BiasAdd", "Add", "Relu"}},
          {FCT::kBiasAddWithGeluApproximate, {"BiasAdd", "GeluApproximate"}},
          {FCT::kRelu, {"Relu"}}};
      OP_REQUIRES_OK(context,
                     InitializeFusedComputation(context, "_ZenMatMul", patterns,
                                                &fused_computation_,
                                                &fused_computation_args_));
    }
  }

  void Compute(OpKernelContext *context) override {
    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-DEF: _ZenMatMul (TF kernel): In Compute!");

    const Tensor &a = context->input(0);
    const Tensor &b = context->input(1);
    // Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(
        context, TensorShapeUtils::IsMatrix(a.shape()),
        errors::InvalidArgument("In[0] is not a matrix. Instead it has shape ",
                                a.shape().DebugString()));
    OP_REQUIRES(
        context, TensorShapeUtils::IsMatrix(b.shape()),
        errors::InvalidArgument("In[1] is not a matrix. Instead it has shape ",
                                b.shape().DebugString()));
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
    dim_pair[0].first = transpose_a_ ? 0 : 1;
    dim_pair[0].second = transpose_b_ ? 1 : 0;

    OP_REQUIRES(context,
                a.dim_size(dim_pair[0].first) == b.dim_size(dim_pair[0].second),
                errors::InvalidArgument("Matrix size-incompatible: In[0]: ",
                                        a.shape().DebugString(),
                                        ", In[1]: ", b.shape().DebugString()));
    int a_dim_remaining = 1 - dim_pair[0].first;
    int b_dim_remaining = 1 - dim_pair[0].second;
    TensorShape out_shape;
    if (is_reshape_) {
      out_shape = {1, a.dim_size(a_dim_remaining), b.dim_size(b_dim_remaining)};
    } else {
      out_shape = {a.dim_size(a_dim_remaining), b.dim_size(b_dim_remaining)};
    }

    bool is_float = std::is_same<T, float>::value;
    // Check for the BF16 support on the machine.
    if (!is_float) {
      bool result = tensorflow::port::TestCPUFeature(
          tensorflow::port::CPUFeature::AVX512F);
      OP_REQUIRES(
          context, result,
          errors::Internal(
              "BF16 AVX512 instruction set is not supported in the machine."));
    }
    // Update the output type.
    ZenTensorType out_type =
        (is_float) ? ZenTensorType::kFloat : ZenTensorType::kBfloat16;

    zendnnEnv zen_env_obj = readEnv();
    Tensor *out = nullptr;
    int zen_enable_mempool =
        zendnn_params_.is_eager ? 0 : zen_env_obj.zenEnableMemPool;
    ZenMemoryPool<T> *zen_pool_buffer = NULL;

    if ((fused_computation_ == FusedComputationType::kBiasAddWithAdd) ||
        (fused_computation_ == FusedComputationType::kBiasAddWithAddAndRelu)) {
      const Tensor &add_tensor = context->input(3);
      context->set_output(0, add_tensor);
      out = context->mutable_output(0);
      if (zen_enable_mempool % MEMPOOL_TYPE) {
        unsigned int thread_id = GetZenTFthreadId(std::this_thread::get_id());
        zen_pool_buffer = ZenMemoryPool<T>::GetZenMemPool(thread_id);
        if (zen_pool_buffer) {
          T *output_array = static_cast<T *>(out->flat<T>().data());
          zen_pool_buffer->ZenMemPoolUpdateTensorPtrStatus(
              context, static_cast<T *>(output_array), zendnn_params_.out_links,
              zendnn_params_.reset);
        }
      }
    } else {
      // ZenMemPool Optimization reuse o/p tensors from the pool. By default its
      // enabled, export ZENDNN_ENABLE_MEMPOOL=0 will disable memory pool
      // optimization.
      // Cases where tensors in pool are not free or requested size is more than
      // available tensor size in Pool, control will fall back to default way of
      // allocation i.e. with allocate_output(..).
      if (zen_enable_mempool % MEMPOOL_TYPE) {
        unsigned int thread_id = GetZenTFthreadId(std::this_thread::get_id());
        zen_pool_buffer = ZenMemoryPool<T>::GetZenMemPool(thread_id);
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
        DataType out_type =
            (is_float) ? DataType::DT_FLOAT : DataType::DT_BFLOAT16;
        // Caching the output buffer and reusing it with persistent tensor.
        int res = cached_buffer_.NumElements();
        Status state = OkStatus();
        if (res <= 0 || res != out_shape.num_elements()) {
          state = context->allocate_temp(out_type, out_shape, &cached_buffer_);
        }
        if (state != OkStatus()) {
          zen_enable_mempool = 0;
        } else {
          out = &cached_buffer_;
          context->set_output(0, *out);
        }
      }
      if (!zen_enable_mempool) {
        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
      }

      if (out->NumElements() == 0) {
        // If a has shape [0, x] or b has shape [x, 0], the output shape
        // is a 0-element matrix, so there is nothing to do.
        return;
      }
    }

    if (a.NumElements() == 0 && b.NumElements() == 0) {
      // If a has shape [x, 0] and b has shape [0, y], the output shape is
      // [x, y] where x and y are non-zero, so we fill the output with zeros.
      // The above condition has not been observed in concerned(ZenDNN) models.
      // Set zero functor is not available directly for plugin, hence it is
      // by-passed here. This implementation can be brought to plugin code in
      // subsequent releases, if necessary.
      if (is_fused) {
        functor::SetZeroFunctor<Device, T> f;
        f(context->eigen_cpu_device(), out->flat<T>());
      }
      return;
    }

    const int m = a.dim_size(1 - dim_pair[0].first);
    const int k = a.dim_size(dim_pair[0].first);
    const int n = b.dim_size(1 - dim_pair[0].second);

    auto a_ptr = const_cast<T *>(a.template flat<T>().data());
    auto b_ptr = const_cast<T *>(b.template flat<T>().data());
    auto c_ptr = (out->template flat<T>().data());

    // Dimensions of matmul source, weights, bias and destination tensors.
    memory::dims src_dims = {m, k};
    memory::dims weight_dims = {k, n};
    memory::dims bias_dims = {1, n};
    memory::dims dst_dims = {m, n};
    memory::format_tag src_format = memory::format_tag::nc;
    memory::format_tag weight_format = (dim_pair[0].second == 1)
                                           ? memory::format_tag::io
                                           : memory::format_tag::oi;

    ZenMatMulParams matmul_params(src_dims, weight_dims, bias_dims, dst_dims,
                                  src_format, weight_format);

    if (!is_fused) {
      T *bias_ptr = NULL;
      if (is_bias_add_gelu) {
        const Tensor &bias = context->input(2);
        bias_ptr = const_cast<T *>(bias.template flat<T>().data());
        matmul_params.post_op_params.push_back({"gelu", {1.0, 0.0, 0.0}});
      }
      ZenMatMulPrimitive<T, T, T, T> *matmul_prim =
          ZenMatMulPrimitiveFactory<T, T, T, T>::Get(matmul_params, 1);
      matmul_prim->Execute(a_ptr, b_ptr, bias_ptr, c_ptr);
    } else {
      bool is_biasadd = (fused_computation_ != FusedComputationType::kRelu);
      LaunchZenFusedMatMulOp<T>()(context, a, b, matmul_params,
                                  fused_computation_, fused_computation_args_,
                                  out, is_biasadd);
    }

    // If ZenMemPool Optimization is enabled(default), update the state of
    // Memory pool based on input_array address.
    if ((zen_env_obj.zenEnableMemPool % MEMPOOL_TYPE) &&
        !zendnn_params_.is_eager && zen_pool_buffer) {
      zen_pool_buffer->ZenMemPoolFree(context, a_ptr);
      zen_pool_buffer->ZenMemPoolFree(context, b_ptr);
    }

    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-DEF: _ZenMatMul (TF kernel): Compute Is Successful!");
  }

 private:
  bool is_reshape_ = false;
  bool transpose_a_ = false;
  bool transpose_b_ = false;
  // TF_GUARDED_BY allows the user to specify a particular mutex that should be
  // held when accessing the annotated variable. GUARDED_VAR indicates that
  // a shared variable is guarded by some unspecified mutex, for use in rare
  // cases where a valid mutex expression cannot be specified.
  //
  // Tensor to hold output buffer memory.
  Tensor cached_buffer_ TF_GUARDED_BY(mu_);

  FusedComputationType fused_computation_ = FusedComputationType::kUndefined;
  FusedComputationArgs fused_computation_args_;

  ZendnnParameters zendnn_params_;
};
#define REGISTER_MATMUL_KERNELS(T)                                             \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_ZenFusedMatMul").Device(DEVICE_CPU).TypeConstraint<T>("T"),       \
      ZenMatMulOp<CPUDevice, T, false, true>);                                 \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_ZenMatMul").Device(DEVICE_CPU).TypeConstraint<T>("T"),            \
      ZenMatMulOp<CPUDevice, T>);                                              \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_ZenMatMulBiasAddGelu").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ZenMatMulOp<CPUDevice, T, true>);                                        \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("MatMulBiasAddGelu").Device(DEVICE_CPU).TypeConstraint<T>("T"),     \
      ZenMatMulOp<CPUDevice, T, true>);

TF_CALL_float(REGISTER_MATMUL_KERNELS);
TF_CALL_bfloat16(REGISTER_MATMUL_KERNELS);
#undef REGISTER_MATMUL_KERNELS

}  // namespace amd_cpu_plugin
