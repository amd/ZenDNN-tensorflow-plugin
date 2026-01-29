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

// Standard headers
#include <iostream>
#include <limits>
#include <memory>
#include <vector>
// TensorFlow plug-in headers
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_conv_kernel.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_kernel_common.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_zendnnl_utils.h"
#include "tensorflow_plugin/src/amd_cpu/util/errors.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_kernel.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_requires.h"
#include "tensorflow_plugin/src/amd_cpu/util/padding.h"
#include "tensorflow_plugin/src/amd_cpu/util/register_types.h"
#include "tensorflow_plugin/src/amd_cpu/util/tensor_format.h"

// ZenDNNL Low Overhead API headers
#include "lowoha_operators/conv/lowoha_conv.hpp"
#include "lowoha_operators/conv/lowoha_conv_common.hpp"

namespace amd_cpu_plugin {

// ZenDNNL Conv implementation using Low Overhead API (conv_direct).
template <typename T>
bool TryExecuteZenDNNLConv2D(
    OpKernelContext* context, const Tensor& input, const Tensor& filter,
    const Tensor* bias, Tensor* output, const Conv2DDimensions& dimensions,
    const Conv2DParameters& params,
    FusedComputationType fusion_type = FusedComputationType::kUndefined,
    const Tensor* addend = nullptr, bool is_depthwise = false) {
  try {
    using namespace zendnnl::lowoha;
    using namespace zendnnl::lowoha::conv;
    using namespace zendnnl::common;

    // Get pointers to TensorFlow tensor data.
    T* input_data = const_cast<T*>(input.flat<T>().data());
    T* filter_data = const_cast<T*>(filter.flat<T>().data());
    T* output_data = output->flat<T>().data();
    T* bias_data = bias ? const_cast<T*>(bias->flat<T>().data()) : nullptr;

    // Setup conv_params structure for Low Overhead API.
    conv_params conv_params;

    conv_params.dims.batch = dimensions.batch;
    conv_params.dims.in_height = dimensions.input_rows;
    conv_params.dims.in_width = dimensions.input_cols;
    conv_params.dims.in_channels = dimensions.in_depth;
    conv_params.dims.filter_height = dimensions.filter_rows;
    conv_params.dims.filter_width = dimensions.filter_cols;
    conv_params.dims.out_channels = dimensions.out_depth;
    conv_params.dims.out_height = dimensions.out_rows;
    conv_params.dims.out_width = dimensions.out_cols;

    conv_params.stride_h = dimensions.stride_rows;
    conv_params.stride_w = dimensions.stride_cols;
    conv_params.pad_top = dimensions.pad_rows_before;
    conv_params.pad_left = dimensions.pad_cols_before;
    conv_params.pad_bottom = dimensions.pad_rows_after;
    conv_params.pad_right = dimensions.pad_cols_after;
    // Check if dilations are available in params before accessing
    if (params.dilations.size() > 2) {
      conv_params.dilation_h = params.dilations[1];
      conv_params.dilation_w = params.dilations[2];
    } else {
      // Default to no dilation if not specified
      conv_params.dilation_h = 1;
      conv_params.dilation_w = 1;
    }
    std::strncpy(conv_params.data_format, "NHWC", 8);

    // Setup depthwise convolution parameters if applicable
    if (is_depthwise) {
      conv_params.depthwise.is_depthwise = true;
      conv_params.depthwise.groups =
          dimensions.in_depth;  // groups = in_channels for depthwise
      // For depthwise: out_channels = in_channels * depth_multiplier
      // So: depth_multiplier = out_channels / in_channels
      if (dimensions.in_depth <= 0) {
        LogZenDNNLInfo("Conv",
                       "Invalid depthwise configuration: in_depth is less than "
                       "or equal to zero");
        return false;
      }
      conv_params.depthwise.depth_multiplier =
          dimensions.out_depth / dimensions.in_depth;

      log_info("DepthwiseConv2D: in_channels=", dimensions.in_depth,
               ", out_channels=", dimensions.out_depth,
               ", depth_multiplier=", conv_params.depthwise.depth_multiplier,
               ", groups=", conv_params.depthwise.groups);
    }

    // Set data types
    if (std::is_same<T, float>::value) {
      conv_params.dtypes.input = data_type_t::f32;
      conv_params.dtypes.filter = data_type_t::f32;
      conv_params.dtypes.output = data_type_t::f32;
      if (bias_data != nullptr) {
        conv_params.dtypes.bias = data_type_t::f32;
      }
    } else if (std::is_same<T, Eigen::bfloat16>::value) {
      conv_params.dtypes.input = data_type_t::bf16;
      conv_params.dtypes.filter = data_type_t::bf16;
      conv_params.dtypes.output = data_type_t::bf16;
      if (bias_data != nullptr) {
        conv_params.dtypes.bias = data_type_t::bf16;
      }
    } else {
      LogZenDNNLInfo("Conv", "Unsupported data type");
      return false;
    }

    // Add post-ops based on fusion type
    using namespace zendnnl::ops;
    using conv_postop = zendnnl::lowoha::conv::conv_postop;
    switch (fusion_type) {
      case FusedComputationType::kBiasAdd:
        // BiasAdd is handled via bias parameter, no additional post-op needed
        log_info("Conv2D ZenDNNL: BiasAdd fusion");
        break;
      case FusedComputationType::kBiasAddWithRelu:
      case FusedComputationType::kRelu: {
        log_info("Conv2D ZenDNNL: BiasAdd+Relu fusion");
        conv_postop relu_po;
        relu_po.po_type = post_op_type_t::relu;
        relu_po.alpha = 0.0f;
        relu_po.beta = 0.0f;
        conv_params.postop_.push_back(relu_po);
        break;
      }
      case FusedComputationType::kBiasAddWithRelu6: {
        log_info("Conv2D ZenDNNL: BiasAdd+Relu6 fusion");
        conv_postop relu6_po;
        relu6_po.po_type = post_op_type_t::clip;
        relu6_po.alpha = 0.0f;
        relu6_po.beta = 6.0f;
        conv_params.postop_.push_back(relu6_po);
        break;
      }
      case FusedComputationType::kBiasAddWithLeakyRelu: {
        log_info("Conv2D ZenDNNL: BiasAdd+LeakyRelu fusion");
        conv_postop leaky_po;
        leaky_po.po_type = post_op_type_t::leaky_relu;
        leaky_po.alpha = 0.2f;  // Default LeakyRelu alpha, can be customized
        leaky_po.beta = 0.0f;
        conv_params.postop_.push_back(leaky_po);
        break;
      }
      case FusedComputationType::kBiasAddWithAdd: {
        log_info("Conv2D ZenDNNL: BiasAdd+Add (residual) fusion");
        if (!addend) {
          LogZenDNNLInfo("Conv2D",
                         "Binary add requested but no addend tensor provided");
          return false;
        }
        // Binary add post-op
        T* addend_data = const_cast<T*>(addend->flat<T>().data());
        conv_postop add_po;
        add_po.po_type = post_op_type_t::binary_add;
        add_po.alpha = 1.0f;  // Scale for residual
        add_po.buff = static_cast<void*>(addend_data);
        add_po.dtype = (std::is_same<T, float>::value) ? data_type_t::f32
                                                       : data_type_t::bf16;
        conv_params.postop_.push_back(add_po);
        break;
      }
      case FusedComputationType::kBiasAddWithAddAndRelu: {
        log_info("Conv2D ZenDNNL: BiasAdd+Add+Relu fusion");
        if (!addend) {
          LogZenDNNLInfo(
              "Conv2D",
              "Binary add+relu requested but no addend tensor provided");
          return false;
        }
        // Binary add post-op first
        T* addend_data = const_cast<T*>(addend->flat<T>().data());
        conv_postop add_po;
        add_po.po_type = post_op_type_t::binary_add;
        add_po.alpha = 1.0f;
        add_po.buff = static_cast<void*>(addend_data);
        add_po.dtype = (std::is_same<T, float>::value) ? data_type_t::f32
                                                       : data_type_t::bf16;
        conv_params.postop_.push_back(add_po);

        // Then relu post-op
        conv_postop relu_po;
        relu_po.po_type = post_op_type_t::relu;
        relu_po.alpha = 0.0f;
        relu_po.beta = 0.0f;
        conv_params.postop_.push_back(relu_po);
        break;
      }
      default:
        // No post-op or unsupported fusion type
        break;
    }

    // Call ZenDNNL Low Overhead API - conv2d_direct
    // TODO: If there is accuracy issue, then we need to set is_weights_const
    // to false.
    status_t status =
        conv_direct(input_data, filter_data, bias_data, output_data,
                    true /* is_weights_const */, conv_params);

    if (status != status_t::success) {
      LogZenDNNLInfo("Conv2D", ("Execution failed with status " +
                                std::to_string(static_cast<int>(status)))
                                   .c_str());
      return false;
    }

    return true;

  } catch (const std::exception& e) {
    LogZenDNNLFallback("Conv2D",
                       ("Exception: " + std::string(e.what())).c_str());
    return false;
  }
}

// Specialized versions for float and bfloat16
template bool TryExecuteZenDNNLConv2D<float>(OpKernelContext*, const Tensor&,
                                             const Tensor&, const Tensor*,
                                             Tensor*, const Conv2DDimensions&,
                                             const Conv2DParameters&,
                                             FusedComputationType,
                                             const Tensor*, bool);
template bool TryExecuteZenDNNLConv2D<Eigen::bfloat16>(
    OpKernelContext*, const Tensor&, const Tensor&, const Tensor*, Tensor*,
    const Conv2DDimensions&, const Conv2DParameters&, FusedComputationType,
    const Tensor*, bool);

template <typename T, bool is_depthwise = false>
class ZenConv2DOp : public OpKernel {
 public:
  explicit ZenConv2DOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, InitConv2DParameters(context, &params_));
    OP_REQUIRES(context, params_.data_format == FORMAT_NHWC,
                errors::Unimplemented("ZenDNN Conv implementation supports "
                                      "NHWC tensor format only for now."));
  }

  void Compute(OpKernelContext* context) override {
    // Old ZenDNN logging removed;

    const Tensor& input = context->input(0);
    const Tensor& filter = context->input(1);
    TensorShape input_shape = input.shape();
    TensorShape filter_shape = filter.shape();

    Conv2DDimensions dimensions;
    ConvUtil conv_util(context, params_, is_depthwise);
    conv_util.InitFwdDimensions(input_shape, filter_shape, &dimensions);

    bool is_input_float = std::is_same<T, float>::value;
    TensorShape out_shape = ShapeFromFormat(
        (params_.data_format), dimensions.batch, dimensions.out_rows,
        dimensions.out_cols, dimensions.out_depth);

    // Output tensor
    Tensor* output = nullptr;

    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    T* input_array = const_cast<T*>(input.template flat<T>().data());
    T* filter_array = const_cast<T*>(filter.template flat<T>().data());
    T* output_array = const_cast<T*>(output->template flat<T>().data());

    T* bias_arr = nullptr;

    // Execute convolution using ZenDNNL (supports both standard and depthwise)
    const Tensor* bias_tensor = nullptr;
    // TODO: Extract bias from context if available

    bool zendnnl_success = TryExecuteZenDNNLConv2D<T>(
        context, input, filter, bias_tensor, output, dimensions, params_,
        FusedComputationType::kUndefined, nullptr, is_depthwise);

    if (zendnnl_success) {
      if (is_depthwise) {
        LogZenDNNLSuccess("DepthwiseConv2D");
      } else {
        LogZenDNNLSuccess("Conv2D");
      }
      return;
    } else {
      LogZenDNNLFallback(is_depthwise ? "DepthwiseConv2D" : "Conv2D", "failed");
      // ZenDNNL execution failed, we MUST report error
      // Output tensor was allocated but not filled with valid data
      OP_REQUIRES(
          context, false,
          errors::Internal(is_depthwise
                               ? "ZenDNNL DepthwiseConv2D execution failed. "
                               : "ZenDNNL Conv2D execution failed. ",
                           "No fallback implementation available. "
                           "Input shape: ",
                           input_shape.DebugString(),
                           ", Filter shape: ", filter_shape.DebugString(),
                           ", Output shape: ", out_shape.DebugString()));
      return;  // Unreachable, but explicit
    }

    // Old ZenDNN logging removed;
  }

 protected:
  Conv2DParameters params_;
  Tensor cached_filter_data_ TF_GUARDED_BY(mu_);
};

#define REGISTER_CONV2D_KERNELS(TYPE)                                  \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("_ZenConv2D").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      ZenConv2DOp<TYPE>);                                              \
  REGISTER_KERNEL_BUILDER(Name("_ZenDepthwiseConv2dNative")            \
                              .Device(DEVICE_CPU)                      \
                              .TypeConstraint<TYPE>("T"),              \
                          ZenConv2DOp<TYPE, true>);

TF_CALL_float(REGISTER_CONV2D_KERNELS);
TF_CALL_bfloat16(REGISTER_CONV2D_KERNELS);

#undef REGISTER_CONV2D_KERNELS

}  // namespace amd_cpu_plugin
