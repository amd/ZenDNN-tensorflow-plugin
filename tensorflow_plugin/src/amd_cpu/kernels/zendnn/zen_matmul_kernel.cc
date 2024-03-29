/*******************************************************************************
 * Modifications Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights
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
                  const FusedComputationArgs &fusion_args, Tensor *output) {
    BiasAddArgs<T> bias_add_args;
    if (BiasAddArgs<T>::IsSupported(fusion)) {
      OP_REQUIRES_OK(context, InitBiasAddArgs(context, &bias_add_args));
    }

    auto a_ptr = const_cast<float *>(a.template flat<T>().data());
    auto b_ptr = const_cast<float *>(b.template flat<T>().data());
    auto c_ptr = (output->template flat<T>().data());
    auto bias_ptr = const_cast<float *>(bias_add_args.bias_add_data);

    switch (fusion) {
      case FusedComputationType::kBiasAdd: {
        ZenMatMulPrimitive<T, T, T, T> *matmul_prim =
            ZenMatMulPrimitiveFactory<T, T, T, T>::Get(matmul_params, 0);
        matmul_prim->Execute(a_ptr, b_ptr, bias_ptr, c_ptr);
        break;
      }
      case FusedComputationType::kBiasAddWithAdd: {
        matmul_params.post_op_params.push_back({"sum", {1.0}});
        ZenMatMulPrimitive<T, T, T, T> *matmul_prim =
            ZenMatMulPrimitiveFactory<T, T, T, T>::Get(matmul_params, 1);
        matmul_prim->Execute(a_ptr, b_ptr, bias_ptr, c_ptr);
        break;
      }
      case FusedComputationType::kBiasAddWithRelu: {
        matmul_params.post_op_params.push_back({"relu", {1.0, 0.0, 0.0}});
        ZenMatMulPrimitive<T, T, T, T> *matmul_prim =
            ZenMatMulPrimitiveFactory<T, T, T, T>::Get(matmul_params, 0);
        matmul_prim->Execute(a_ptr, b_ptr, bias_ptr, c_ptr);
        break;
      }
      case FusedComputationType::kBiasAddWithAddAndRelu: {
        matmul_params.post_op_params.push_back({"sum", {1.0}});
        matmul_params.post_op_params.push_back({"relu", {1.0, 0.0, 0.0}});
        ZenMatMulPrimitive<T, T, T, T> *matmul_prim =
            ZenMatMulPrimitiveFactory<T, T, T, T>::Get(matmul_params, 1);
        matmul_prim->Execute(a_ptr, b_ptr, bias_ptr, c_ptr);
        break;
      }
      case FusedComputationType::kBiasAddWithGeluApproximate: {
        matmul_params.post_op_params.push_back(
            {"GeluApproximate", {1.0, 1.0, 0.0}});
        ZenMatMulPrimitive<T, T, T, T> *matmul_prim =
            ZenMatMulPrimitiveFactory<T, T, T, T>::Get(matmul_params, 1);
        matmul_prim->Execute(a_ptr, b_ptr, bias_ptr, c_ptr);
        break;
      }
      case FusedComputationType::kBiasAddWithGeluExact: {
        matmul_params.post_op_params.push_back({"GeluExact", {1.0, 1.0, 0.0}});
        ZenMatMulPrimitive<T, T, T, T> *matmul_prim =
            ZenMatMulPrimitiveFactory<T, T, T, T>::Get(matmul_params, 1);
        matmul_prim->Execute(a_ptr, b_ptr, bias_ptr, c_ptr);
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
    if (is_fused) {
      using FCT = FusedComputationType;
      patterns = {
          {FCT::kBiasAdd, {"BiasAdd"}},
          {FCT::kBiasAddWithAdd, {"BiasAdd", "Add"}},
          {FCT::kBiasAddWithElu, {"BiasAdd", "Elu"}},
          {FCT::kBiasAddWithRelu, {"BiasAdd", "Relu"}},
          {FCT::kBiasAddWithRelu6, {"BiasAdd", "Relu6"}},
          {FCT::kBiasAddWithGeluExact, {"BiasAdd", "GeluExact"}},
          {FCT::kBiasAddWithAddAndRelu, {"BiasAdd", "Add", "Relu"}},
          {FCT::kBiasAddWithGeluApproximate, {"BiasAdd", "GeluApproximate"}}};
      OP_REQUIRES_OK(context,
                     InitializeFusedComputation(context, "_ZenMatMul", patterns,
                                                &fused_computation_,
                                                &fused_computation_args_));
    }
  }

  void Compute(OpKernelContext *context) override {
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
    TensorShape out_shape(
        {a.dim_size(a_dim_remaining), b.dim_size(b_dim_remaining)});

    // Update the output type.
    ZenTensorType out_type = ZenTensorType::kFloat;

    zendnnEnv zen_env_obj = readEnv();
    Tensor *out = nullptr;
    int zen_enable_mempool = zen_env_obj.zenEnableMemPool &&
                             !zendnn_params_.is_eager &&
                             context->expected_output_dtype(0) == DT_FLOAT;
    ZenMemoryPool<T> *zen_pool_buffer = NULL;

    if ((fused_computation_ == FusedComputationType::kBiasAddWithAdd) ||
        (fused_computation_ == FusedComputationType::kBiasAddWithAddAndRelu)) {
      const Tensor &add_tensor = context->input(3);
      context->set_output(0, add_tensor);
      out = context->mutable_output(0);
      if (zen_enable_mempool) {
        unsigned int thread_id = GetZenTFthreadId(std::this_thread::get_id());
        zen_pool_buffer = ZenMemoryPool<T>::GetZenMemPool(thread_id);
        if (zen_pool_buffer) {
          T *output_array = static_cast<T *>(out->flat<T>().data());
          zen_pool_buffer->ZenMemPoolUpdateTensorPtrStatus(
              context, static_cast<float *>(output_array),
              zendnn_params_.out_links, zendnn_params_.reset);
        }
      }
    } else {
      // ZenMemPool Optimization reuse o/p tensors from the pool. By default its
      // enabled, export ZENDNN_ENABLE_MEMPOOL=0 will disable memory pool
      // optimization.
      // Cases where tensors in pool are not free or requested size is more than
      // available tensor size in Pool, control will fall back to default way of
      // allocation i.e. with allocate_output(..).
      if (zen_enable_mempool) {
        unsigned int thread_id = GetZenTFthreadId(std::this_thread::get_id());
        zen_pool_buffer = ZenMemoryPool<T>::GetZenMemPool(thread_id);
        if (zen_pool_buffer) {
          int status = zen_pool_buffer->AcquireZenPoolTensor(
              context, &out, out_shape, zendnn_params_.out_links,
              zendnn_params_.reset, out_type);
          if (status) {
            zen_enable_mempool = false;
          }
        } else {
          zen_enable_mempool = false;
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
      // functor::SetZeroFunctor<Device, T> f;
      // f(context->eigen_device<Device>(), out->flat<T>());
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
    memory::dims weight_dims = {n, k};
    memory::dims bias_dims = {n};
    memory::dims dst_dims = {m, n};
    memory::format_tag src_format = memory::format_tag::nc;
    memory::format_tag weight_format = (dim_pair[0].second == 1)
                                           ? memory::format_tag::oi
                                           : memory::format_tag::io;

    ZenMatMulParams matmul_params(src_dims, weight_dims, bias_dims, dst_dims,
                                  src_format, weight_format);

    if (!is_fused) {
      float *bias_ptr = NULL;
      if (is_bias_add_gelu) {
        const Tensor &bias = context->input(2);
        bias_ptr = const_cast<T *>(bias.template flat<T>().data());
        matmul_params.post_op_params.push_back({"gelu", {1.0, 0.0, 0.0}});
      }
      ZenMatMulPrimitive<T, T, T, T> *matmul_prim =
          ZenMatMulPrimitiveFactory<T, T, T, T>::Get(matmul_params, 0);
      matmul_prim->Execute(a_ptr, b_ptr, bias_ptr, c_ptr);
    } else {
      LaunchZenFusedMatMulOp<T>()(context, a, b, matmul_params,
                                  fused_computation_, fused_computation_args_,
                                  out);
    }

    // If ZenMemPool Optimization is enabled(default), update the state of
    // Memory pool based on input_array address.
    if (zen_env_obj.zenEnableMemPool && !zendnn_params_.is_eager &&
        (a.dtype() == DT_FLOAT && b.dtype() == DT_FLOAT) && zen_pool_buffer) {
      zen_pool_buffer->ZenMemPoolFree(context, a_ptr);
      zen_pool_buffer->ZenMemPoolFree(context, b_ptr);
    }
  }

 private:
  bool transpose_a_;
  bool transpose_b_;

  FusedComputationType fused_computation_ = FusedComputationType::kUndefined;
  FusedComputationArgs fused_computation_args_;

  ZendnnParameters zendnn_params_;
};

REGISTER_KERNEL_BUILDER(
    Name("_ZenFusedMatMul").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    ZenMatMulOp<CPUDevice, float, false, true>);
REGISTER_KERNEL_BUILDER(
    Name("_ZenMatMul").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    ZenMatMulOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("_ZenMatMulBiasAddGelu").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    ZenMatMulOp<CPUDevice, float, true>);
REGISTER_KERNEL_BUILDER(
    Name("MatMulBiasAddGelu").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    ZenMatMulOp<CPUDevice, float, true>);

}  // namespace amd_cpu_plugin
