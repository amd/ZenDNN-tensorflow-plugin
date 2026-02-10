/*******************************************************************************
 * Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
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
#include <iostream>
#include <limits>
#include <memory>
#include <vector>
// TensorFlow plug-in headers.
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_conv_kernel.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_kernel_common.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_zendnnl_utils.h"
#include "tensorflow_plugin/src/amd_cpu/util/errors.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_kernel.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_requires.h"
#include "tensorflow_plugin/src/amd_cpu/util/padding.h"
#include "tensorflow_plugin/src/amd_cpu/util/register_types.h"
#include "tensorflow_plugin/src/amd_cpu/util/tensor_format.h"
// ZenDNNL logging support
#include "common/zendnnl_global.hpp"

namespace amd_cpu_plugin {

// Forward declaration - TryExecuteZenDNNLConv2D is defined in
// zen_conv2d_kernel.cc
template <typename T>
bool TryExecuteZenDNNLConv2D(OpKernelContext *context, const Tensor &input,
                             const Tensor &filter, const Tensor *bias,
                             Tensor *output, const Conv2DDimensions &dimensions,
                             const Conv2DParameters &params,
                             FusedComputationType fusion_type,
                             const Tensor *addend, bool is_depthwise = false);

template <typename T, bool is_depthwise = false>
class ZenFusedConv2DOp : public OpKernel {
 public:
  explicit ZenFusedConv2DOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, InitConv2DParameters(context, &params_));
    using FCT = FusedComputationType;

    std::vector<FusedComputationPattern> patterns = {};
    patterns = {
        {FCT::kBiasAdd, {"BiasAdd"}},
        {FCT::kBiasAddWithRelu, {"BiasAdd", "Relu"}},
        {FCT::kBiasAddWithRelu6, {"BiasAdd", "Relu6"}},
        {FCT::kBiasAddWithLeakyRelu, {"BiasAdd", "LeakyRelu"}},
        {FCT::kBiasAddWithAdd, {"BiasAdd", "Add"}},
        {FCT::kBiasAddWithAddAndRelu, {"BiasAdd", "Add", "Relu"}},
        // TODO (plugin): Add back the fusion for Conv2D. Once it is supported.
        // {FCT::kFusedBatchNorm, {"FusedBatchNorm"}},
        // {FCT::kFusedBatchNormWithRelu, {"FusedBatchNorm", "Relu"}},
        // {FCT::kFusedBatchNormWithLeakyRelu, {"FusedBatchNorm", "LeakyRelu"}},
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

  void Compute(OpKernelContext *context) override {
    zendnnl::error_handling::apilog_info(
        "Executing _ZenFusedConv2D Compute, is_depthwise=", is_depthwise);

    const Tensor &input = context->input(0);
    const Tensor &filter = context->input(1);
    TensorShape input_shape = input.shape();
    TensorShape filter_shape = filter.shape();

    Conv2DDimensions dimensions;
    ConvUtil conv_util(context, params_, is_depthwise);
    conv_util.InitFwdDimensions(input_shape, filter_shape, &dimensions);

    TensorShape out_shape = ShapeFromFormat(
        (params_.data_format), dimensions.batch, dimensions.out_rows,
        dimensions.out_cols, dimensions.out_depth);

    // Output tensor.
    Tensor *output = nullptr;

    if ((fused_computation_ == FusedComputationType::kBiasAddWithAdd) ||
        (fused_computation_ == FusedComputationType::kBiasAddWithAddAndRelu)) {
      const Tensor &add_tensor = context->input(3);
      context->set_output(0, add_tensor);
      output = context->mutable_output(0);
    } else {
      OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
    }

    // Extract bias tensor
    const Tensor *bias = nullptr;
    if (fused_computation_ != FusedComputationType::kRelu) {
      bias = &context->input(2);  // BiasAdd is typically input(2)
    }

    // Extract addend tensor for residual connections
    const Tensor *addend = nullptr;
    if (fused_computation_ == FusedComputationType::kBiasAddWithAdd ||
        fused_computation_ == FusedComputationType::kBiasAddWithAddAndRelu) {
      addend = &context->input(3);  // Add tensor is input(3)
    }

    // Execute using ZenDNNL (supports both standard and depthwise)
    bool zendnnl_success = TryExecuteZenDNNLConv2D<T>(
        context, input, filter, bias, output, dimensions, params_,
        fused_computation_, addend, is_depthwise);

    if (zendnnl_success) {
      if (is_depthwise) {
        LogZenDNNLSuccess("_ZenFusedDepthwiseConv2dNative");
      } else {
        LogZenDNNLSuccess("_ZenFusedConv2D");
      }
    } else {
      LogZenDNNLFallback(
          is_depthwise ? "_ZenFusedDepthwiseConv2dNative" : "_ZenFusedConv2D",
          "failed");
    }

    zendnnl::error_handling::apilog_info("_ZenFusedConv2D Compute completed");
  }

 private:
  Conv2DParameters params_;
  float alpha_ = 0.0;
  Tensor cached_filter_data_ TF_GUARDED_BY(mu_);
  FusedComputationType fused_computation_ = FusedComputationType::kUndefined;
  FusedComputationArgs fused_computation_args_;
};

#define REGISTER_FUSED_CONV2D_KERNELS(TYPE)                                 \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("_ZenFusedConv2D").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      ZenFusedConv2DOp<TYPE>);                                              \
  REGISTER_KERNEL_BUILDER(Name("_ZenFusedDepthwiseConv2dNative")            \
                              .Device(DEVICE_CPU)                           \
                              .TypeConstraint<TYPE>("T"),                   \
                          ZenFusedConv2DOp<TYPE, true>);

TF_CALL_float(REGISTER_FUSED_CONV2D_KERNELS);
TF_CALL_bfloat16(REGISTER_FUSED_CONV2D_KERNELS);

#undef REGISTER_FUSED_CONV2D_KERNELS

}  // namespace amd_cpu_plugin
