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

// Standard headers
#include <iostream>
#include <limits>
#include <memory>
#include <vector>
// TensorFlow plug-in headers
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_conv_kernel.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_conv_kernel_fused.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_kernel_common.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_mempool.h"
#include "tensorflow_plugin/src/amd_cpu/util/errors.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_kernel.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_requires.h"
#include "tensorflow_plugin/src/amd_cpu/util/padding.h"
#include "tensorflow_plugin/src/amd_cpu/util/register_types.h"
#include "tensorflow_plugin/src/amd_cpu/util/tensor_format.h"

namespace amd_cpu_plugin {

template <typename T, bool pad_enabled = false, bool is_depthwise = false,
          bool is_sum = false>
class ZenFusedConv2DOp : public OpKernel {
 public:
  explicit ZenFusedConv2DOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, InitConv2DParameters(context, &params_));
    OP_REQUIRES_OK(context, InitZendnnParameters(context, &zendnn_params_));
    using FCT = FusedComputationType;

    std::vector<FusedComputationPattern> patterns;
    patterns = {
        {FCT::kBiasAdd, {"BiasAdd"}},
        {FCT::kBiasAddWithRelu, {"BiasAdd", "Relu"}},
        {FCT::kBiasAddWithRelu6, {"BiasAdd", "Relu6"}},
        {FCT::kBiasAddWithElu, {"BiasAdd", "Elu"}},
        {FCT::kBiasAddWithLeakyRelu, {"BiasAdd", "LeakyRelu"}},
        {FCT::kBiasAddWithAdd, {"BiasAdd", "Add"}},
        {FCT::kBiasAddWithAddAndRelu, {"BiasAdd", "Add", "Relu"}},
        {FCT::kFusedBatchNorm, {"FusedBatchNorm"}},
        {FCT::kFusedBatchNormWithRelu, {"FusedBatchNorm", "Relu"}},
        {FCT::kFusedBatchNormWithRelu6, {"FusedBatchNorm", "Relu6"}},
        {FCT::kFusedBatchNormWithElu, {"FusedBatchNorm", "Elu"}},
        {FCT::kFusedBatchNormWithLeakyRelu, {"FusedBatchNorm", "LeakyRelu"}},
    };

    OP_REQUIRES_OK(context, InitializeFusedComputation(
                                context, "_ZenConv2D", patterns,
                                &fused_computation_, &fused_computation_args_));
    if (fused_computation_ == FCT::kBiasAddWithLeakyRelu ||
        fused_computation_ == FCT::kFusedBatchNormWithLeakyRelu) {
      OP_REQUIRES_OK(context, context->GetAttr("leakyrelu_alpha", &alpha_));
    } else {
      // LeakyRelu fusion not found. Setting leakyrelu alpha to 0.0f in all
      // other cases when the alpha will not be used because it is only
      // connected to LeakyRelu activation.
      alpha_ = 0.0f;
    }
  }

  void Compute(OpKernelContext* context) override {
    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-DEF: _ZenFusedConv (TF kernel): In Compute!");

    const Tensor& input = context->input(0);
    const Tensor& filter = context->input(1);
    TensorShape input_shape = input.shape();
    TensorShape filter_shape = filter.shape();

    const Tensor& dinput = is_sum ? context->input(6) : input;

    Conv2DDimensions dimensions;
    ConvUtil conv_util(context, params_, is_depthwise);
    conv_util.InitFwdDimensions(input_shape, filter_shape, &dimensions);

    // Update the output type.
    bool is_input_float = std::is_same<T, float>::value;
    ZenTensorType out_type =
        (is_input_float) ? ZenTensorType::kFloat : ZenTensorType::kBfloat16;

    TensorShape out_shape = ShapeFromFormat(
        (params_.data_format), dimensions.batch, dimensions.out_rows,
        dimensions.out_cols, dimensions.out_depth);

    // Output tensor
    Tensor* output = nullptr;
    zendnnEnv zen_env_obj = readEnv();

    int zen_enable_mempool =
        zendnn_params_.is_eager ? 0 : zen_env_obj.zenEnableMemPool;
    ZenMemoryPool<T>* zen_pool_buffer = NULL;

    if ((fused_computation_ == FusedComputationType::kBiasAddWithAdd) ||
        (fused_computation_ == FusedComputationType::kBiasAddWithAddAndRelu)) {
      const Tensor& add_tensor = context->input(3);
      context->set_output(0, add_tensor);
      output = context->mutable_output(0);
      if (zen_enable_mempool % MEMPOOL_TYPE) {
        unsigned int thread_id = GetZenTFthreadId(std::this_thread::get_id());
        zen_pool_buffer = ZenMemoryPool<T>::GetZenMemPool(thread_id);
        if (zen_pool_buffer) {
          T* output_array = static_cast<T*>(output->flat<T>().data());
          zen_pool_buffer->ZenMemPoolUpdateTensorPtrStatus(
              context, static_cast<T*>(output_array), zendnn_params_.out_links,
              zendnn_params_.reset);
        }
      }
    } else {
      // ZenMemPool Optimization reuse o/p tensors from the pool. By default its
      // enabled, export ZENDNN_ENABLE_MEMPOOL=0 will disable memory pool
      // optimization.
      // Cases where tensors in pool are not free or requested size is more than
      // available tensor size in Pool, control will fall back to default way of
      // allocation i.e. with allocate_output(..)
      if (zen_enable_mempool % MEMPOOL_TYPE) {
        unsigned int thread_id = GetZenTFthreadId(std::this_thread::get_id());
        zen_pool_buffer = ZenMemoryPool<T>::GetZenMemPool(thread_id);
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
      }
      if (!(zen_enable_mempool % MEMPOOL_TYPE)) {
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, out_shape, &output));
      }
    }

    if (is_sum) {
      LaunchZenFusedConv2DSumOp<T>()(
          context, input, filter, dinput, fused_computation_,
          fused_computation_args_, dimensions, output, zendnn_params_.is_eager,
          zendnn_params_.reorder_before, zendnn_params_.reorder_after,
          &cached_filter_data_);
    } else {
      LaunchZenFusedConv2DOp<T>()(
          context, input, filter, fused_computation_, fused_computation_args_,
          dimensions, output, zendnn_params_.is_eager,
          zendnn_params_.reorder_before, zendnn_params_.reorder_after,
          &cached_filter_data_, is_depthwise, alpha_);
    }

    // If ZenMemPool Optimization is enabled(default), update the state of
    // memory pool based on input_array address.
    if ((zen_env_obj.zenEnableMemPool % MEMPOOL_TYPE) &&
        !zendnn_params_.is_eager && zen_pool_buffer) {
      T* input_array = const_cast<T*>(input.template flat<T>().data());
      zen_pool_buffer->ZenMemPoolFree(context,
                                      reinterpret_cast<void*>(input_array));
      if (is_sum) {
        T* dinput_array = const_cast<T*>(dinput.template flat<T>().data());
        zen_pool_buffer->ZenMemPoolFree(context,
                                        reinterpret_cast<void*>(dinput_array));
      }
    }

    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-DEF: _ZenFusedConv (TF kernel): Compute Is Successful!");
  }

 private:
  Conv2DParameters params_;
  float alpha_;
  Tensor cached_filter_data_ TF_GUARDED_BY(mu_);
  FusedComputationType fused_computation_ = FusedComputationType::kUndefined;
  FusedComputationArgs fused_computation_args_;

  /* ZenDNN specific */
  ZendnnParameters zendnn_params_;
};

#define REGISTER_FUSED_CONV2D_KERNELS(TYPE)                                 \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("_ZenFusedConv2D").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      ZenFusedConv2DOp<TYPE>);                                              \
  REGISTER_KERNEL_BUILDER(Name("_ZenFusedDepthwiseConv2dNative")            \
                              .Device(DEVICE_CPU)                           \
                              .TypeConstraint<TYPE>("T"),                   \
                          ZenFusedConv2DOp<TYPE, false, true, false>);

TF_CALL_float(REGISTER_FUSED_CONV2D_KERNELS);
TF_CALL_bfloat16(REGISTER_FUSED_CONV2D_KERNELS);

#undef REGISTER_FUSED_CONV2D_KERNELS

REGISTER_KERNEL_BUILDER(
    Name("_ZenFusedConv2DSum").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    ZenFusedConv2DOp<float, false, false, true>);

}  // namespace amd_cpu_plugin
