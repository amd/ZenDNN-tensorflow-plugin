/*******************************************************************************
 * Modifications Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights
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
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_mempool.h"
#include "tensorflow_plugin/src/amd_cpu/util/errors.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_kernel.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_requires.h"
#include "tensorflow_plugin/src/amd_cpu/util/padding.h"
#include "tensorflow_plugin/src/amd_cpu/util/register_types.h"
#include "tensorflow_plugin/src/amd_cpu/util/tensor_format.h"

namespace amd_cpu_plugin {

#define NEW_API 1
#if NEW_API
void ZenGemmConvolution2D(void* input_array, int batch_size, int channels,
                          int height, int width, void* filter_array,
                          int output_channels, int kernel_h, int kernel_w,
                          float pad_t, float pad_l, float pad_b, float pad_r,
                          int stride_h, int stride_w, void* bias_array,
                          void* output_array, int out_height, int out_width,
                          bool relu_fused, bool batchnorm_fused, bool add_fused,
                          void* bn_scale, void* bn_mean, void* bn_offset);

void ZenConvolution2DBiasOrRelu(
    zendnn::engine eng, zendnn::stream s, zendnn::primitive_attr conv_attr,
    void* input_array, int batch_size, int channels, int height, int width,
    void* filter_array, int output_channels, int kernel_h, int kernel_w,
    float pad_t, float pad_l, float pad_b, float pad_r, int stride_h,
    int stride_w, void* bias_array, void* output_array, int out_height,
    int out_width, bool is_eager, bool reorder_before, bool reorder_after,
    void* cached_filter_data_, void* context);
#endif

template <typename T, bool pad_enabled = false, bool is_depthwise = false>
class ZenConv2DOp : public OpKernel {
 public:
  explicit ZenConv2DOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, InitConv2DParameters(context, &params_));
    OP_REQUIRES(context, params_.data_format == FORMAT_NHWC,
                errors::Unimplemented("ZenDNN Conv implementation supports "
                                      "NHWC tensor format only for now."));
    OP_REQUIRES_OK(context, InitZendnnParameters(context, &zendnn_params_));
  }

  void Compute(OpKernelContext* context) override {
    zendnnInfo(ZENDNN_FWKLOG, "ZEN-OP-DEF: _ZenConv (TF kernel): In Compute!");

    const Tensor& input = context->input(0);
    const Tensor& filter = context->input(1);
    TensorShape input_shape = input.shape();
    TensorShape filter_shape = filter.shape();

    Conv2DDimensions dimensions;
    ConvUtil conv_util(context, params_, is_depthwise);
    conv_util.InitFwdDimensions(input_shape, filter_shape, &dimensions);

    // Update the output type.
    ZenTensorType out_type = ZenTensorType::kFloat;

    TensorShape out_shape = ShapeFromFormat(
        (params_.data_format), dimensions.batch, dimensions.out_rows,
        dimensions.out_cols, dimensions.out_depth);

    // Output tensor
    Tensor* output = nullptr;
    zendnnEnv zen_env_obj = readEnv();
    bool blocked = zen_env_obj.zenConvAlgo == zenConvAlgoType::DIRECT1 &&
                   !zendnn_params_.is_eager;
    bool blocked_nhwc = zen_env_obj.zenConvAlgo == zenConvAlgoType::DIRECT2;

    if (dimensions.out_depth % 8 != 0 && blocked && !blocked_nhwc) {
      OP_REQUIRES_OK(context,
                     errors::Internal(
                         "ZENDNN_BLOCKED_FORMAT not supported for this model, "
                         "Please use another data format."));
    }

    int zen_enable_mempool = zen_env_obj.zenEnableMemPool &&
                             !zendnn_params_.is_eager &&
                             context->expected_output_dtype(0) == DT_FLOAT;
    ZenMemoryPool<T>* zen_pool_buffer = NULL;

    // ZenMemPool Optimization reuse o/p tensors from the pool. By default its
    // enabled, export ZENDNN_ENABLE_MEMPOOL=0 to disable memory pool
    // optimization.
    // Cases where tensors in pool are not free or requested size is more than
    // available tensor size in Pool, control will fall back to default way of
    // allocation i.e. with allocate_output(..).
    // ZenMempool Optimization is not supported by Depthwise Convolution due to
    // performance drop.
    if (zen_enable_mempool) {
      unsigned int thread_id = GetZenTFthreadId(std::this_thread::get_id());
      zen_pool_buffer = ZenMemoryPool<T>::GetZenMemPool(thread_id);
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
    }
    if (!zen_enable_mempool) {
      OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
    }

    T* input_array = const_cast<T*>(input.template flat<T>().data());
    T* filter_array = const_cast<T*>(filter.template flat<T>().data());
    T* output_array = const_cast<T*>(output->template flat<T>().data());

#if NEW_API
    float* bias_arr = nullptr;

    // Direct convolution.
    primitive_attr conv_attr;
    ZenExecutor* ex = ex->getInstance();
    engine eng = ex->getEngine();
    stream s = ex->getStream();
    if (is_depthwise) {
      ZenConvolution2DDepthwise(
          eng, s, conv_attr, input_array, dimensions.batch, dimensions.in_depth,
          dimensions.input_rows, dimensions.input_cols, filter_array,
          dimensions.out_depth, dimensions.filter_rows, dimensions.filter_cols,
          dimensions.pad_rows_before, dimensions.pad_cols_before,
          dimensions.pad_rows_after, dimensions.pad_cols_after,
          dimensions.stride_rows, dimensions.stride_cols, bias_arr,
          output_array, dimensions.out_rows, dimensions.out_cols,
          zendnn_params_.is_eager, zendnn_params_.reorder_before,
          zendnn_params_.reorder_after, &(cached_filter_data_), context);
    } else if (blocked || blocked_nhwc) {
      ZenConvolution2DBiasOrRelu(
          eng, s, conv_attr, input_array, dimensions.batch, dimensions.in_depth,
          dimensions.input_rows, dimensions.input_cols, filter_array,
          dimensions.out_depth, dimensions.filter_rows, dimensions.filter_cols,
          dimensions.pad_rows_before, dimensions.pad_cols_before,
          dimensions.pad_rows_after, dimensions.pad_cols_after,
          dimensions.stride_rows, dimensions.stride_cols, bias_arr,
          output_array, dimensions.out_rows, dimensions.out_cols,
          zendnn_params_.is_eager, zendnn_params_.reorder_before,
          zendnn_params_.reorder_after, (&cached_filter_data_), context);
    } else {
      // GEMM based convolution.
      ZenGemmConvolution2D(
          input_array, dimensions.batch, dimensions.in_depth,
          dimensions.input_rows, dimensions.input_cols, filter_array,
          dimensions.out_depth, dimensions.filter_rows, dimensions.filter_cols,
          dimensions.pad_rows_before, dimensions.pad_cols_before,
          dimensions.pad_rows_after, dimensions.pad_cols_after,
          dimensions.stride_rows, dimensions.stride_cols, bias_arr,
          output_array, dimensions.out_rows, dimensions.out_cols, false, false,
          false, nullptr, nullptr, nullptr);
    }
#else
    zenConvolution2D(input_array, dimensions.batch, dimensions.in_depth,
                     dimensions.input_rows, dimensions.input_cols, filter_array,
                     dimensions.out_depth, dimensions.filter_rows,
                     dimensions.filter_cols, dimensions.pad_rows_before,
                     dimensions.pad_cols_before, dimensions.pad_rows_after,
                     dimensions.pad_cols_after, dimensions.stride_rows,
                     dimensions.stride_cols, output_array, dimensions.out_rows,
                     dimensions.out_cols);
#endif

    // If ZenMemPool Optimization is enabled(default), update the state of
    // Memory pool based on input_array address.
    if (zen_env_obj.zenEnableMemPool && !zendnn_params_.is_eager &&
        (input.dtype() == DT_FLOAT) && zen_pool_buffer) {
      T* input_array = const_cast<T*>(input.template flat<T>().data());
      zen_pool_buffer->ZenMemPoolFree(context,
                                      reinterpret_cast<float*>(input_array));
    }

    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-DEF: _ZenConv (TF kernel): Compute Is Successful!");
  }

 protected:
  Conv2DParameters params_;
  Tensor cached_filter_data_ TF_GUARDED_BY(mu_);

  /* ZenDNN specific */
  ZendnnParameters zendnn_params_;
};

REGISTER_KERNEL_BUILDER(
    Name("_ZenConv2D").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    ZenConv2DOp<float>);

REGISTER_KERNEL_BUILDER(Name("_ZenDepthwiseConv2dNative")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        ZenConv2DOp<float, false, true>);

}  // namespace amd_cpu_plugin
