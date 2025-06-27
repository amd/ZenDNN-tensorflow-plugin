/*******************************************************************************
 * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 ******************************************************************************/

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_CONV_KERNEL_FUSED_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_CONV_KERNEL_FUSED_H_

// Standard headers.
#include <iostream>
#include <string>
#include <vector>
// TensorFlow plug-in headers.
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/fused_eigen_output_kernels.h"
#include "tensorflow_plugin/src/amd_cpu/util/zen_utils.h"

namespace amd_cpu_plugin {

void ZenGemmConvolution2D(void *input_array, int batch_size, int channels,
                          int height, int width, void *filter_array,
                          int output_channels, int kernel_h, int kernel_w,
                          float pad_t, float pad_l, float pad_b, float pad_r,
                          int stride_h, int stride_w, void *bias_array,
                          void *output_array, int out_height, int out_width,
                          bool relu_fused, bool batchnorm_fused, bool add_fused,
                          void *bn_scale, void *bn_mean, void *bn_offset,
                          const float ops_alpha = 0.0f);

void ZenQuantizedConv2DBiasOrRelu(
    zendnn::engine eng, zendnn::stream s, zendnn::primitive_attr conv_attr,
    void *context, void *input_array, int batch_size, int channels, int height,
    int width, void *filter_array, int output_channels, int kernel_h,
    int kernel_w, int out_height, int out_width, int pad_t, int pad_l,
    int pad_b, int pad_r, int stride_h, int stride_w, void *bias_array,
    const std::vector<float> &scale, void *output_array, void *output_min,
    void *output_max, bool Tinput, bool Toutput, bool Tbias,
    const std::vector<float> &bias_scale, bool is_relu, bool is_sum,
    bool is_signed, float factor, int depth, float scale_output,
    float scale_summand, void *cached_filter_data_, bool reset);

template <typename T>
void ZenBlockedConv2DBiasEltSum(
    zendnn::engine eng, zendnn::stream s, zendnn::primitive_attr conv_attr,
    void *input_array, int batch_size, int channels, int height, int width,
    void *filter_array, int output_channels, int kernel_h, int kernel_w,
    int pad_t, int pad_l, int pad_b, int pad_r, int stride_h, int stride_w,
    void *bias_array, void *output_array, int out_height, int out_width,
    bool is_eager, bool reorder_before, bool reorder_after,
    void *cached_filter_data_, void *context);

template <typename T>
void ZenConvolution2DBiasOrRelu(
    zendnn::engine eng, zendnn::stream s, zendnn::primitive_attr conv_attr,
    void *input_array, int batch_size, int channels, int height, int width,
    void *filter_array, int output_channels, int kernel_h, int kernel_w,
    float pad_t, float pad_l, float pad_b, float pad_r, int stride_h,
    int stride_w, void *bias_array, void *output_array, int out_height,
    int out_width, bool is_eager, bool reorder_before, bool reorder_after,
    void *cached_filter_data_, void *context);

void ZenConvolution2DBatchNormOrRelu(
    zendnn::engine eng, zendnn::stream s, zendnn::primitive_attr conv_attr,
    void *input_array, int batch_size, int channels, int height, int width,
    void *filter_array, int output_channels, int kernel_h, int kernel_w,
    float pad_t, float pad_l, float pad_b, float pad_r, int stride_h,
    int stride_w, void *bias_array, void *batch_norm_scale,
    void *batch_norm_mean, void *batch_norm_offset, void *elementwise_input,
    void *output_array, int out_height, int out_width, bool relu_fused,
    bool batchnorm_fused, bool is_eager, bool reorder_before,
    bool reorder_after, void *cached_filter_data_, void *context,
    const float ops_alpha = 0.0f);

template <typename T>
struct LaunchZenFusedConv2DOp {
  void operator()(OpKernelContext *context, const Tensor &input,
                  const Tensor &filter, const FusedComputationType fusion,
                  const FusedComputationArgs &fusion_args,
                  const Conv2DDimensions &dimensions, Tensor *output,
                  bool is_eager, bool reorder_before, bool reorder_after,
                  Tensor *cached_filter_data_, bool is_depthwise, float alpha) {
    OP_REQUIRES(context, dimensions.in_depth == filter.dim_size(2),
                errors::Unimplemented("Fused conv implementation does not "
                                      "support grouped convolutions for now."));

    const Tensor &bias = context->input(2);
    if (BiasAddArgs<T>::IsSupported(fusion)) {
      for (int i = 0; i < bias.dims() - 1; i++) {
        OP_REQUIRES(
            context, bias.dim_size(i) == 1,
            errors::InvalidArgument("For bias_dims > 1, all except the "
                                    "last dimension (channel) must be 1, got: ",
                                    bias.shape().DebugString()));
      }
    }

    FusedBatchNormArgs<T> fused_batch_norm_args;
    if (FusedBatchNormArgs<T>::IsSupported(fusion)) {
      OP_REQUIRES_OK(context,
                     InitFusedBatchNormArgs(context, fusion_args.epsilon,
                                            &fused_batch_norm_args, &alpha));
    }

    T *input_array = const_cast<T *>(input.template flat<T>().data());
    T *filter_array = const_cast<T *>(filter.template flat<T>().data());
    T *output_array = const_cast<T *>(output->template flat<T>().data());

    zendnnEnv zen_env_obj = readEnv();
    bool blocked_nhwc = zen_env_obj.zenConvAlgo == zenConvAlgoType::DIRECT1;

    ZenExecutor *ex = ex->getInstance();
    engine eng = ex->getEngine();
    stream s = ex->getStream();

    bool is_input_float = std::is_same<T, float>::value;

    if (!(is_depthwise || blocked_nhwc)) {
      OP_REQUIRES(context, is_input_float,
                  errors::Unimplemented(
                      "ZenDNN GEMM path only supported for FP32 data type"));
    }

    switch (fusion) {
      case FusedComputationType::kUndefined:
        OP_REQUIRES_OK(context, errors::Internal("Fusion type is undefined"));
        break;
      case FusedComputationType::kBiasAdd: {
        T *bias_arr = const_cast<T *>(bias.flat<T>().data());
        primitive_attr conv_attr;
        if (is_depthwise) {
          ZenConvolution2DDepthwise<T>(
              eng, s, conv_attr, input_array, dimensions.batch,
              dimensions.in_depth, dimensions.input_rows, dimensions.input_cols,
              filter_array, dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, is_eager,
              reorder_before, reorder_after, cached_filter_data_, context);
        } else if (blocked_nhwc) {
          // Direct convolution.
          ZenConvolution2DBiasOrRelu<T>(
              eng, s, conv_attr, input_array, dimensions.batch,
              dimensions.in_depth, dimensions.input_rows, dimensions.input_cols,
              filter_array, dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, is_eager,
              reorder_before, reorder_after, cached_filter_data_, context);
        } else {
          // GEMM based convolution.
          ZenGemmConvolution2D(
              input_array, dimensions.batch, dimensions.in_depth,
              dimensions.input_rows, dimensions.input_cols, filter_array,
              dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, false, false, false,
              nullptr, nullptr, nullptr);
        }
        break;
      }
      case FusedComputationType::kBiasAddWithRelu: {
        T *bias_arr = const_cast<T *>(bias.flat<T>().data());
        primitive_attr conv_attr;
        // Configure post-ops.
        const float ops_scale = 1.f;
        const float ops_alpha = 0.f;  // relu negative slope.
        const float ops_beta = 0.f;
        post_ops ops;
        ops.append_eltwise(ops_scale, algorithm::eltwise_relu, ops_alpha,
                           ops_beta);
        conv_attr.set_post_ops(ops);
        if (is_depthwise) {
          ZenConvolution2DDepthwise<T>(
              eng, s, conv_attr, input_array, dimensions.batch,
              dimensions.in_depth, dimensions.input_rows, dimensions.input_cols,
              filter_array, dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, is_eager,
              reorder_before, reorder_after, cached_filter_data_, context);
        } else if (blocked_nhwc) {
          // Direct convolution.
          ZenConvolution2DBiasOrRelu<T>(
              eng, s, conv_attr, input_array, dimensions.batch,
              dimensions.in_depth, dimensions.input_rows, dimensions.input_cols,
              filter_array, dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, is_eager,
              reorder_before, reorder_after, cached_filter_data_, context);
        } else {
          // GEMM based convolution.
          ZenGemmConvolution2D(
              input_array, dimensions.batch, dimensions.in_depth,
              dimensions.input_rows, dimensions.input_cols, filter_array,
              dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, true, false, false,
              nullptr, nullptr, nullptr);
        }
        break;
      }
      case FusedComputationType::kBiasAddWithRelu6: {
        T *bias_arr = const_cast<T *>(bias.flat<T>().data());
        primitive_attr conv_attr;
        // Configure post-ops.
        const float ops_scale = 1.f;
        const float ops_alpha = 6.0;  // relu negative slope.
        const float ops_beta = 0.f;
        post_ops ops;
        ops.append_eltwise(ops_scale, algorithm::eltwise_bounded_relu,
                           ops_alpha, ops_beta);
        conv_attr.set_post_ops(ops);
        if (is_depthwise) {
          ZenConvolution2DDepthwise<T>(
              eng, s, conv_attr, input_array, dimensions.batch,
              dimensions.in_depth, dimensions.input_rows, dimensions.input_cols,
              filter_array, dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, is_eager,
              reorder_before, reorder_after, cached_filter_data_, context);
        } else if (blocked_nhwc) {
          // Direct convolution.
          ZenConvolution2DBiasOrRelu<T>(
              eng, s, conv_attr, input_array, dimensions.batch,
              dimensions.in_depth, dimensions.input_rows, dimensions.input_cols,
              filter_array, dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, is_eager,
              reorder_before, reorder_after, cached_filter_data_, context);
        } else {
          // GEMM based convolution.
          ZenGemmConvolution2D(
              input_array, dimensions.batch, dimensions.in_depth,
              dimensions.input_rows, dimensions.input_cols, filter_array,
              dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, true, false, false,
              nullptr, nullptr, nullptr);
          zenClipOp(zen_env_obj, reinterpret_cast<float *>(output_array), 6.0F,
                    dimensions.batch * dimensions.out_depth *
                        dimensions.out_rows * dimensions.out_cols);
        }
        break;
      }
      case FusedComputationType::kBiasAddWithLeakyRelu: {
        T *bias_arr = const_cast<T *>(bias.flat<T>().data());
        primitive_attr conv_attr;
        // Configure post-ops.
        const float ops_scale = 1.f;
        const float ops_alpha = alpha;  // relu negative slope.
        const float ops_beta = 0.f;
        post_ops ops;
        ops.append_eltwise(ops_scale, algorithm::eltwise_bounded_relu,
                           ops_alpha, ops_beta);
        conv_attr.set_post_ops(ops);
        if (is_depthwise) {
          ZenConvolution2DDepthwise<T>(
              eng, s, conv_attr, input_array, dimensions.batch,
              dimensions.in_depth, dimensions.input_rows, dimensions.input_cols,
              filter_array, dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, is_eager,
              reorder_before, reorder_after, cached_filter_data_, context);
        } else if (blocked_nhwc) {
          // Direct convolution.
          ZenConvolution2DBiasOrRelu<T>(
              eng, s, conv_attr, input_array, dimensions.batch,
              dimensions.in_depth, dimensions.input_rows, dimensions.input_cols,
              filter_array, dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, is_eager,
              reorder_before, reorder_after, cached_filter_data_, context);
        } else {
          // GEMM based convolution.
          ZenGemmConvolution2D(
              input_array, dimensions.batch, dimensions.in_depth,
              dimensions.input_rows, dimensions.input_cols, filter_array,
              dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, true, false, false,
              nullptr, nullptr, nullptr, ops_alpha);
        }
        break;
      }
      case FusedComputationType::kBiasAddWithElu:
        OP_REQUIRES_OK(context, errors::Internal("Fusion type not supported"));
        break;
      case FusedComputationType::kBiasAddWithAdd: {
        T *bias_arr = const_cast<T *>(bias.flat<T>().data());
        if (blocked_nhwc) {
          // Direct convolution.
          primitive_attr conv_attr;
          // Configure post-ops.
          float ops_scale = 1.0;
          post_ops ops;
          ops.append_sum(ops_scale);
          conv_attr.set_post_ops(ops);
          // Configure post-ops.
          ZenBlockedConv2DBiasEltSum<T>(
              eng, s, conv_attr, input_array, dimensions.batch,
              dimensions.in_depth, dimensions.input_rows, dimensions.input_cols,
              filter_array, dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, is_eager,
              reorder_before, reorder_after, cached_filter_data_, context);
        } else {
          // GEMM based convolution.
          ZenGemmConvolution2D(
              input_array, dimensions.batch, dimensions.in_depth,
              dimensions.input_rows, dimensions.input_cols, filter_array,
              dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, false, false, true,
              nullptr, nullptr, nullptr);
        }
        break;
      }
      case FusedComputationType::kBiasAddWithAddAndRelu: {
        T *bias_arr = const_cast<T *>(bias.flat<T>().data());
        if (blocked_nhwc) {
          // Direct convolution.
          primitive_attr conv_attr;
          // Configure post-ops.
          const float ops_scale = 1.f;
          const float ops_alpha = 0.f;  // relu negative slope.
          const float ops_beta = 0.f;
          post_ops ops;
          ops.append_sum(ops_scale);
          ops.append_eltwise(ops_scale, algorithm::eltwise_relu, ops_alpha,
                             ops_beta);
          conv_attr.set_post_ops(ops);
          // Configure post-ops.
          ZenBlockedConv2DBiasEltSum<T>(
              eng, s, conv_attr, input_array, dimensions.batch,
              dimensions.in_depth, dimensions.input_rows, dimensions.input_cols,
              filter_array, dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, is_eager,
              reorder_before, reorder_after, cached_filter_data_, context);
        } else {
          // GEMM based convolution.
          ZenGemmConvolution2D(
              input_array, dimensions.batch, dimensions.in_depth,
              dimensions.input_rows, dimensions.input_cols, filter_array,
              dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, true, false, true,
              nullptr, nullptr, nullptr);
        }
        break;
      }
      case FusedComputationType::kFusedBatchNorm: {
        T *bias_arr = NULL;
        T *batch_norm_mean_data =
            const_cast<T *>(fused_batch_norm_args.estimated_mean_data);
        T *batch_norm_offset_data =
            const_cast<T *>(fused_batch_norm_args.offset_data);
        if (blocked_nhwc) {
          primitive_attr conv_attr;
          ZenConvolution2DBatchNormOrRelu(
              eng, s, conv_attr, input_array, dimensions.batch,
              dimensions.in_depth, dimensions.input_rows, dimensions.input_cols,
              filter_array, dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr,
              fused_batch_norm_args.scaling_factor.data(), batch_norm_mean_data,
              batch_norm_offset_data,
              NULL,  // elementwise_input is not required.
              output_array, dimensions.out_rows, dimensions.out_cols, false,
              true, is_eager, reorder_before, reorder_after,
              cached_filter_data_, context);
        } else {
          // GEMM based convolution.
          ZenGemmConvolution2D(
              input_array, dimensions.batch, dimensions.in_depth,
              dimensions.input_rows, dimensions.input_cols, filter_array,
              dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, false, true, false,
              fused_batch_norm_args.scaling_factor.data(), batch_norm_mean_data,
              batch_norm_offset_data);
        }
        break;
      }
      case FusedComputationType::kFusedBatchNormWithRelu: {
        T *bias_arr = NULL;
        T *batch_norm_mean_data =
            const_cast<T *>(fused_batch_norm_args.estimated_mean_data);
        T *batch_norm_offset_data =
            const_cast<T *>(fused_batch_norm_args.offset_data);
        if (blocked_nhwc) {
          primitive_attr conv_attr;
          ZenConvolution2DBatchNormOrRelu(
              eng, s, conv_attr, input_array, dimensions.batch,
              dimensions.in_depth, dimensions.input_rows, dimensions.input_cols,
              filter_array, dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr,
              fused_batch_norm_args.scaling_factor.data(), batch_norm_mean_data,
              batch_norm_offset_data,
              NULL,  // elementwise_input is not required.
              output_array, dimensions.out_rows, dimensions.out_cols, true,
              true, is_eager, reorder_before, reorder_after,
              cached_filter_data_, context);
        } else {
          // GEMM based convolution.
          ZenGemmConvolution2D(
              input_array, dimensions.batch, dimensions.in_depth,
              dimensions.input_rows, dimensions.input_cols, filter_array,
              dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, true, true, false,
              fused_batch_norm_args.scaling_factor.data(), batch_norm_mean_data,
              batch_norm_offset_data);
        }
        break;
      }
      case FusedComputationType::kFusedBatchNormWithRelu6:
        OP_REQUIRES_OK(context, errors::Internal("Fusion type not supported"));
        break;
      case FusedComputationType::kFusedBatchNormWithElu:
        OP_REQUIRES_OK(context, errors::Internal("Fusion type not supported"));
        break;
      case FusedComputationType::kFusedBatchNormWithLeakyRelu: {
        T *bias_arr = NULL;
        T *batch_norm_mean_data =
            const_cast<T *>(fused_batch_norm_args.estimated_mean_data);
        T *batch_norm_offset_data =
            const_cast<T *>(fused_batch_norm_args.offset_data);
        const float ops_alpha = fused_batch_norm_args.leakyrelu_alpha;
        if (blocked_nhwc) {
          primitive_attr conv_attr;
          ZenConvolution2DBatchNormOrRelu(
              eng, s, conv_attr, input_array, dimensions.batch,
              dimensions.in_depth, dimensions.input_rows, dimensions.input_cols,
              filter_array, dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr,
              fused_batch_norm_args.scaling_factor.data(), batch_norm_mean_data,
              batch_norm_offset_data,
              NULL,  // The elementwise_input is not required.
              output_array, dimensions.out_rows, dimensions.out_cols, true,
              true, is_eager, reorder_before, reorder_after,
              cached_filter_data_, context, ops_alpha);
        } else {
          // GEMM based convolution.
          ZenGemmConvolution2D(
              input_array, dimensions.batch, dimensions.in_depth,
              dimensions.input_rows, dimensions.input_cols, filter_array,
              dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, true, true, false,
              fused_batch_norm_args.scaling_factor.data(), batch_norm_mean_data,
              batch_norm_offset_data, ops_alpha);
        }
        break;
      }
      default:
        OP_REQUIRES_OK(context, errors::Internal("Fusion type not supported"));
        break;
    }
  }
};

template <typename T>
struct LaunchZenFusedConv2DSumOp {
  void operator()(OpKernelContext *context, const Tensor &input,
                  const Tensor &filter, const Tensor &dinput,
                  const FusedComputationType fusion,
                  const FusedComputationArgs &fusion_args,
                  const Conv2DDimensions &dimensions, Tensor *output,
                  bool is_eager, bool reorder_before, bool reorder_after,
                  Tensor *cached_filter_data_) {
    bool is_input_float = std::is_same<T, float>::value;
    OP_REQUIRES(context, is_input_float,
                errors::Unimplemented(
                    "_ZenFusedConv2DSum only supported for FP32 data type"));
    OP_REQUIRES(context, dimensions.in_depth == filter.dim_size(2),
                errors::Unimplemented("Fused conv implementation does not "
                                      "support grouped convolutions for now."));

    FusedBatchNormArgs<float> fused_batch_norm_args;
    if (FusedBatchNormArgs<float>::IsSupported(fusion)) {
      OP_REQUIRES_OK(context,
                     InitFusedBatchNormArgs(context, fusion_args.epsilon,
                                            &fused_batch_norm_args));
    }

    float *input_array =
        const_cast<float *>(input.template flat<float>().data());
    float *dinput_array =
        const_cast<float *>(dinput.template flat<float>().data());
    float *filter_array =
        const_cast<float *>(filter.template flat<float>().data());
    float *output_array =
        const_cast<float *>(output->template flat<float>().data());

    float *bia_arr = NULL;
    float *batch_norm_mean_data =
        const_cast<float *>(fused_batch_norm_args.estimated_mean_data);
    float *batch_norm_offset_data =
        const_cast<float *>(fused_batch_norm_args.offset_data);

    primitive_attr conv_attr;

    zendnnEnv zen_env_obj = readEnv();
    bool blocked_nhwc = zen_env_obj.zenConvAlgo == zenConvAlgoType::DIRECT1;

    if (blocked_nhwc) {
      ZenExecutor *ex = ex->getInstance();
      engine eng = ex->getEngine();
      stream s = ex->getStream();
      ZenConvolution2DBatchNormOrRelu(
          eng, s, conv_attr, input_array, dimensions.batch, dimensions.in_depth,
          dimensions.input_rows, dimensions.input_cols, filter_array,
          dimensions.out_depth, dimensions.filter_rows, dimensions.filter_cols,
          dimensions.pad_rows_before, dimensions.pad_cols_before,
          dimensions.pad_rows_after, dimensions.pad_cols_after,
          dimensions.stride_rows, dimensions.stride_cols, bia_arr,
          fused_batch_norm_args.scaling_factor.data(), batch_norm_mean_data,
          batch_norm_offset_data, dinput_array, output_array,
          dimensions.out_rows, dimensions.out_cols, true, true, is_eager,
          reorder_before, reorder_after, cached_filter_data_, context);
    } else {
      // TODO(zendnn): This else part will go once NEW API is supported for
      // non-blocked format.
      zenConvolution2DwithBatchNormsum(
          input_array, dimensions.batch, dimensions.in_depth,
          dimensions.input_rows, dimensions.input_cols, filter_array,
          dimensions.out_depth, dimensions.filter_rows, dimensions.filter_cols,
          dimensions.pad_rows_before, dimensions.pad_cols_before,
          dimensions.pad_rows_after, dimensions.pad_cols_after,
          dimensions.stride_rows, dimensions.stride_cols,
          fused_batch_norm_args.scaling_factor.data(),
          fused_batch_norm_args.estimated_mean_data,
          fused_batch_norm_args.offset_data, dinput_array, output_array,
          dimensions.out_rows, dimensions.out_cols);
    }
  }
};

}  // namespace amd_cpu_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_CONV_KERNEL_FUSED_H_
