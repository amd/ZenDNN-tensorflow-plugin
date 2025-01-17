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
#include <iostream>
#include <limits>
#include <memory>
#include <vector>
// TensorFlow plug-in headers.
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

template <typename Tinput, typename Tfilter, typename Tbias, typename Toutput,
          typename Ttemp_output, bool bias_enabled, bool is_depthwise,
          bool is_relu, bool is_sum, bool is_signed>
class ZenQuantizedConv2DOp : public OpKernel {
 public:
  explicit ZenQuantizedConv2DOp(OpKernelConstruction *context)
      : OpKernel(context) {
    // Support for new padding definition for Quantized Conv ops in TF.
    // Fix for pad accuracy issue with ResNet50v1.5 model.
    // For Quantized Conv ops, there is no EXPLICIT pad type (as in FP32 Conv
    // ops). A new attribute padding_list is used with VALID pad type. If
    // padding_list has custom values, then it should be used. If no custom
    // values have been defined, then pad value of 0 is used (VALID type).
    if (context->HasAttr("padding_list")) {
      OP_REQUIRES_OK(context, context->GetAttr("padding_list", &padding_list_));
    }
    OP_REQUIRES_OK(context, InitConv2DParameters(context, &params_));
    OP_REQUIRES_OK(context, InitZendnnParameters(context, &zendnn_params_));
  }
  void Compute(OpKernelContext *context) override {
    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-DEF: _ZenQuantizedConv2D (TF kernel): In Compute!");

    ZenExecutor *ex = ex->getInstance();
    engine eng = ex->getEngine();
    stream s = ex->getStream();

    // Input Filter and Bias.
    const Tensor &input = context->input(0);
    Tinput *input_array = static_cast<Tinput *>(
        const_cast<Tinput *>(input.flat<Tinput>().data()));

    const Tensor &filter = context->input(1);
    Tfilter *filter_array = static_cast<Tfilter *>(
        const_cast<Tfilter *>(filter.flat<Tfilter>().data()));

    const Tensor &bias = context->input(2);
    for (int i = 0; i < bias.dims() - 1; i++) {
      OP_REQUIRES(
          context, bias.dim_size(i) == 1,
          errors::InvalidArgument("For bias_dims > 1, all except the "
                                  "last dimension (channel) must be 1, got: ",
                                  bias.shape().DebugString()));
    }
    Tbias *bias_array = const_cast<Tbias *>(bias.flat<Tbias>().data());

    // Update the output type.
    ZenTensorType out_type = ZenTensorType::kQint8;
    if (std::is_same<Toutput, quint8>::value) {
      out_type = ZenTensorType::kQuint8;
    }
    bool bias_type = std::is_same<Tbias, qint32>::value;
    // Input type is defined (signed or unsigned).
    // Fix for ResNet50v1.5 INT8 model where signed INT8 input is used.
    bool in_type = std::is_same<Tinput, quint8>::value;

    TensorShape zen_out_shape_max, zen_out_shape_min;

    // Compute Convolution/Quantization Specific parameters.

    Tensor *output = nullptr, *output_min = nullptr, *output_max = nullptr;
    TensorShape out_shape;
    int batch_size, channels, height, width, output_channels, kernel_height,
        kernel_width;
    int bias_index_offset = bias_enabled ? 1 : 0;
    float scale_output = 0.0, scale_summand = 0.0;

    const int stride_rows =
        GetTensorDim(params_.strides, params_.data_format, 'H');
    const int stride_cols =
        GetTensorDim(params_.strides, params_.data_format, 'W');
    const int dilation_rows =
        GetTensorDim(params_.dilations, params_.data_format, 'H');
    const int dilation_cols =
        GetTensorDim(params_.dilations, params_.data_format, 'W');

    int64 out_rows = 0, out_cols = 0;
    int64 pad_rows_before = 0, pad_rows_after = 0, pad_cols_before = 0,
          pad_cols_after = 0;

    batch_size = input.dim_size(0);
    channels = input.dim_size(3);

    if (!is_depthwise) {
      kernel_width = filter.dim_size(1);
      kernel_height = filter.dim_size(0);
      output_channels = filter.dim_size(3);
    } else {
      kernel_width = filter.dim_size(1);
      kernel_height = filter.dim_size(0);
      output_channels = filter.dim_size(2);
    }

    height = input.dim_size(1);
    width = input.dim_size(2);

    GetWindowedOutputSizeVerboseV2(width, kernel_width, dilation_cols,
                                   stride_cols, params_.padding, &out_cols,
                                   &pad_cols_before, &pad_cols_after);
    GetWindowedOutputSizeVerboseV2(height, kernel_height, dilation_rows,
                                   stride_rows, params_.padding, &out_rows,
                                   &pad_rows_before, &pad_rows_after);

    // Support for new padding type in Quantized Conv ops.
    for (auto const &padding_val : padding_list_) {
      if (padding_val > 0) {
        pad_rows_before = pad_rows_after = pad_cols_before = pad_cols_after =
            padding_val;
        out_rows = out_cols =
            (height + pad_cols_before + pad_cols_after - kernel_height) /
                (stride_rows) +
            1;
        break;
      }
    }

    out_shape = ShapeFromFormat(params_.data_format, batch_size, out_rows,
                                out_cols, output_channels);

    OP_REQUIRES_OK(context,
                   context->allocate_output(1, zen_out_shape_min, &output_min));
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, zen_out_shape_max, &output_max));

    const float min_input =
        context->input(2 + bias_index_offset).flat<float>()(0);
    const float max_input =
        context->input(3 + bias_index_offset).flat<float>()(0);
    const Tensor &min_filter_vector = context->input(4 + bias_index_offset);
    const Tensor &max_filter_vector = context->input(5 + bias_index_offset);
    const float min_freezed_output =
        context->input(6 + bias_index_offset).flat<float>()(0);
    const float max_freezed_output =
        context->input(7 + bias_index_offset).flat<float>()(0);

    output_min->flat<float>()(0) =
        context->input(6 + bias_index_offset).flat<float>()(0);
    output_max->flat<float>()(0) =
        context->input(7 + bias_index_offset).flat<float>()(0);

    output_min = context->mutable_output(1);
    output_max = context->mutable_output(2);

    float factor = is_signed ? 127.0f : 255.0f;
    float ftype = (bool)out_type ? 255.0f : 127.0f;
    size_t depth = 1;
    depth = min_filter_vector.NumElements();
    const float *min_filter = min_filter_vector.flat<float>().data();
    const float *max_filter = max_filter_vector.flat<float>().data();
    std::vector<float> scales(depth);
    std::vector<float> bias_scales(depth);
    float input_range = std::max(std::abs(min_input), std::abs(max_input));
    float output_range =
        std::max(std::abs(min_freezed_output), std::abs(max_freezed_output));

    for (size_t i = 0; i < depth; ++i) {
      float filter_range =
          std::max(std::abs(min_filter[i]), std::abs(max_filter[i]));
      // Changes to fix accuracy issue with ResNet50v1.5 first Conv layer.
      const float int_const_scale_limit =
          (in_type) ? 255.0 * 127.0 : 127.0 * 127.0;
      scales[i] = (ftype * input_range * filter_range) /
                  (int_const_scale_limit * output_range);
      bias_scales[i] = int_const_scale_limit /
                       (input_range * std::max(std::abs(min_filter[i]),
                                               std::abs(max_filter[i])));
    }

    zendnnEnv zen_env_obj = readEnv();
    int zen_enable_mempool =
        zendnn_params_.is_eager ? 0 : zen_env_obj.zenEnableMemPool;
    ZenMemoryPool<Toutput> *zen_pool_buffer = NULL;

    if (is_sum) {
      const float min_freezed_summand =
          context->input(9 + bias_index_offset).flat<float>()(0);
      const float max_freezed_summand =
          context->input(10 + bias_index_offset).flat<float>()(0);
      scale_output =
          std::max(std::abs(min_freezed_output), std::abs(max_freezed_output));
      scale_summand = std::max(std::abs(min_freezed_summand),
                               std::abs(max_freezed_summand));

      Tensor &add_tensor = const_cast<Tensor &>(context->input(9));

      OP_REQUIRES_OK(context, add_tensor.BitcastFrom(add_tensor, DT_QUINT8,
                                                     add_tensor.shape()));
      context->set_output(0, add_tensor);
      output = context->mutable_output(0);
      if (zen_enable_mempool % MEMPOOL_TYPE) {
        unsigned int thread_id = GetZenTFthreadId(std::this_thread::get_id());
        zen_pool_buffer = ZenMemoryPool<Toutput>::GetZenMemPool(thread_id);
        if (zen_pool_buffer) {
          const void *output_array;
          if (out_type == ZenTensorType::kQint8) {
            output_array = const_cast<qint8 *>(output->flat<qint8>().data());
            // Quantized models have 3 outputs. 1 output is used
            // for computation, other 2 outputs are used during dequantize.
            zen_pool_buffer->ZenMemPoolUpdateTensorPtrStatus(
                context, (qint8 *)output_array, zendnn_params_.out_links - 2,
                zendnn_params_.reset);
          } else if (out_type == ZenTensorType::kQuint8) {
            output_array = const_cast<quint8 *>(output->flat<quint8>().data());
            zen_pool_buffer->ZenMemPoolUpdateTensorPtrStatus(
                context, (quint8 *)output_array, zendnn_params_.out_links - 2,
                zendnn_params_.reset);
          }
        }
      }
    } else {
      if (zen_enable_mempool % MEMPOOL_TYPE) {
        unsigned int thread_id = GetZenTFthreadId(std::this_thread::get_id());
        zen_pool_buffer = ZenMemoryPool<Toutput>::GetZenMemPool(thread_id);
        if (zen_pool_buffer) {
          // Quantized models have 3 outputs. 1 output is used
          // for computation, other 2 outputs are used during dequantize.
          int status = zen_pool_buffer->AcquireZenPoolTensor(
              context, &output, out_shape, zendnn_params_.out_links - 2,
              zendnn_params_.reset, out_type);
          if (status) {
            zen_enable_mempool = 0;
          }
        } else {
          zen_enable_mempool = 0;
        }
      }
      if (!(zen_enable_mempool % MEMPOOL_TYPE)) {
        // Outtype is not required for default allocation because context
        // maintains allocation data Type for outputs.
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, out_shape, &output));
      }
    }

    auto output_map = output->tensor<Toutput, 4>();
    void *output_array = const_cast<Toutput *>(output_map.data());

    // There are edge cases where destination memory type from registered Op is
    // unsigned but results of the Operations are signed. Example patterns is
    // when convolution does not have relu and postops is signed. In these cases
    // post memory allocation we cast them to signed based on the new out_type.
    // TODO: Hardcoding to be removed with alternative patch.
    if (depth == 1) {
      if (!is_relu || is_signed) {
        out_type = ZenTensorType::kQint8;
      }
    } else {
      out_type = ZenTensorType::kQuint8;
    }
    // Accuracy fix for ResNet50v1.5 INT8 model.
    // TODO(plugin): Add an alternative fix.
    // For specific Convolution layers, output type is unsigned instead of
    // signed. 7 Convolution layers involved in this fix.
    if ((channels == 64 && output_channels == 256 && height == 56) ||
        (channels == 256 && output_channels == 512 && height == 56) ||
        (channels == 128 && output_channels == 512 && height == 28) ||
        (channels == 512 && output_channels == 1024 && height == 28) ||
        (channels == 256 && output_channels == 1024 && height == 14) ||
        (channels == 1024 && output_channels == 2048 && height == 14) ||
        (channels == 512 && output_channels == 2048 && height == 7)) {
      out_type = ZenTensorType::kQint8;
    }

    primitive_attr conv_attr;

    ZenQuantizedConv2DBiasOrRelu(
        eng, s, conv_attr, context, input_array, batch_size, channels, height,
        width, filter_array, output_channels, kernel_height, kernel_width,
        out_rows, out_cols, pad_rows_before, pad_cols_before, pad_rows_after,
        pad_cols_after, stride_rows, stride_cols, bias_array, scales,
        output_array, output_min, output_max, in_type, (bool)out_type,
        bias_type, bias_scales, is_relu, is_sum, is_signed, factor,
        is_depthwise, scale_output, scale_summand, &cached_filter_data_,
        zendnn_params_.reset);
    if ((zen_env_obj.zenEnableMemPool % MEMPOOL_TYPE) && zen_pool_buffer) {
      zen_pool_buffer->ZenMemPoolFree(context, (void *)input_array);
    }
    zendnnInfo(
        ZENDNN_FWKLOG,
        "ZEN-OP-DEF: _ZenQuantizedConv2D (TF kernel): Compute Is Successful!");
  }

 private:
  Conv2DParameters params_;
  // Additional attributes to support new Padding definition and tensors.
  std::vector<int64> padding_list_;
  // ZenDNN specific.
  ZendnnParameters zendnn_params_;
  Tensor cached_filter_data_ TF_GUARDED_BY(mu_);
  Tensor cached_data_ TF_GUARDED_BY(mu_);
};

template <typename T, bool pad_enabled = false, bool is_depthwise = false,
          bool is_sum = false>
class ZenFusedConv2DOp : public OpKernel {
 public:
  explicit ZenFusedConv2DOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, InitConv2DParameters(context, &params_));
    OP_REQUIRES_OK(context, InitZendnnParameters(context, &zendnn_params_));
    using FCT = FusedComputationType;

    std::vector<FusedComputationPattern> patterns = {};
    patterns = {
        {FCT::kBiasAdd, {"BiasAdd"}},
        {FCT::kBiasAddWithRelu, {"BiasAdd", "Relu"}},
        {FCT::kBiasAddWithRelu6, {"BiasAdd", "Relu6"}},
        {FCT::kBiasAddWithLeakyRelu, {"BiasAdd", "LeakyRelu"}},
        {FCT::kBiasAddWithAdd, {"BiasAdd", "Add"}},
        {FCT::kBiasAddWithAddAndRelu, {"BiasAdd", "Add", "Relu"}},
        {FCT::kFusedBatchNorm, {"FusedBatchNorm"}},
        {FCT::kFusedBatchNormWithRelu, {"FusedBatchNorm", "Relu"}},
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

  void Compute(OpKernelContext *context) override {
    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-DEF: _ZenFusedConv (TF kernel): In Compute!");

    const Tensor &input = context->input(0);
    const Tensor &filter = context->input(1);
    TensorShape input_shape = input.shape();
    TensorShape filter_shape = filter.shape();

    const Tensor &dinput = is_sum ? context->input(6) : input;

    Conv2DDimensions dimensions;
    ConvUtil conv_util(context, params_, is_depthwise);
    conv_util.InitFwdDimensions(input_shape, filter_shape, &dimensions);

    // Update the output type.
    bool is_input_float = std::is_same<T, float>::value;
    ZenTensorType out_type =
        (is_input_float) ? ZenTensorType::kFloat : ZenTensorType::kBfloat16;
    DataType dtype =
        (is_input_float) ? DataType::DT_FLOAT : DataType::DT_BFLOAT16;

    TensorShape out_shape = ShapeFromFormat(
        (params_.data_format), dimensions.batch, dimensions.out_rows,
        dimensions.out_cols, dimensions.out_depth);

    // Output tensor.
    Tensor *output = nullptr;
    zendnnEnv zen_env_obj = readEnv();

    int zen_enable_mempool =
        zendnn_params_.is_eager ? 0 : zen_env_obj.zenEnableMemPool;
    ZenMemoryPool<T> *zen_pool_buffer = NULL;

    if ((fused_computation_ == FusedComputationType::kBiasAddWithAdd) ||
        (fused_computation_ == FusedComputationType::kBiasAddWithAddAndRelu)) {
      const Tensor &add_tensor = context->input(3);
      context->set_output(0, add_tensor);
      output = context->mutable_output(0);
      if (zen_enable_mempool % MEMPOOL_TYPE) {
        unsigned int thread_id = GetZenTFthreadId(std::this_thread::get_id());
        zen_pool_buffer = ZenMemoryPool<T>::GetZenMemPool(thread_id);
        if (zen_pool_buffer) {
          T *output_array = static_cast<T *>(output->flat<T>().data());
          zen_pool_buffer->ZenMemPoolUpdateTensorPtrStatus(
              context, static_cast<T *>(output_array), zendnn_params_.out_links,
              zendnn_params_.reset);
        }
      }
    } else {
      // ZenMemPool Optimization reuse o/p tensors from the pool. By default
      // it's enabled, export ZENDNN_ENABLE_MEMPOOL=0 will disable memory pool
      // optimization.
      // Cases where tensors in pool are not free or requested size is more than
      // available tensor size in Pool, control will fall back to default way of
      // allocation i.e. with allocate_output(..).
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
      } else if (zen_enable_mempool) {
        // Caching the output buffer and reusing it with persistent tensor.
        int res = cached_buffer_.NumElements();
        Status state = OkStatus();
        if (res <= 0 || res != out_shape.num_elements()) {
          state = context->allocate_temp(dtype, out_shape, &cached_buffer_);
        }
        if (state != OkStatus()) {
          zen_enable_mempool = 0;
        } else {
          output = &cached_buffer_;
          context->set_output(0, *output);
        }
      }
      if (!zen_enable_mempool) {
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
      T *input_array = const_cast<T *>(input.template flat<T>().data());
      zen_pool_buffer->ZenMemPoolFree(context,
                                      reinterpret_cast<void *>(input_array));
      if (is_sum) {
        T *dinput_array = const_cast<T *>(dinput.template flat<T>().data());
        zen_pool_buffer->ZenMemPoolFree(context,
                                        reinterpret_cast<void *>(dinput_array));
      }
    }

    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-DEF: _ZenFusedConv (TF kernel): Compute Is Successful!");
  }

 private:
  Conv2DParameters params_;
  float alpha_ = 0.0;
  // TF_GUARDED_BY allows the user to specify a particular mutex that should be
  // held when accessing the annotated variable. GUARDED_VAR indicates that
  // a shared variable is guarded by some unspecified mutex, for use in rare
  // cases where a valid mutex expression cannot be specified.
  //
  // Tensor to hold output buffer memory.
  Tensor cached_buffer_ TF_GUARDED_BY(mu_);
  Tensor cached_filter_data_ TF_GUARDED_BY(mu_);
  FusedComputationType fused_computation_ = FusedComputationType::kUndefined;
  FusedComputationArgs fused_computation_args_;

  // ZenDNN specific.
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
REGISTER_KERNEL_BUILDER(
    Name("_ZenQuantizedConv2DWithBiasAndReluAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<quint8>("out_type")
        .TypeConstraint<qint32>("Tbias"),
    ZenQuantizedConv2DOp<quint8, qint8, qint32, quint8, qint8, true, false,
                         true, false, false>);
REGISTER_KERNEL_BUILDER(
    Name("_ZenQuantizedConv2DWithBiasAndReluAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<quint8>("out_type")
        .TypeConstraint<float>("Tbias"),
    ZenQuantizedConv2DOp<quint8, qint8, float, quint8, qint8, true, false, true,
                         false, false>);

REGISTER_KERNEL_BUILDER(Name("_ZenQuantizedConv2DWithBiasAndReluAndRequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<quint8>("out_type")
                            .TypeConstraint<float>("Tbias"),
                        ZenQuantizedConv2DOp<qint8, qint8, float, quint8, qint8,
                                             true, false, true, false, false>);
REGISTER_KERNEL_BUILDER(
    Name("_ZenQuantizedConv2DWithBiasAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint32>("Tbias")
        .TypeConstraint<qint8>("out_type"),
    ZenQuantizedConv2DOp<quint8, qint8, qint32, qint8, qint8, true, false,
                         false, false, false>);
REGISTER_KERNEL_BUILDER(Name("_ZenQuantizedConv2DWithBiasAndRequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<float>("Tbias")
                            .TypeConstraint<qint8>("out_type"),
                        ZenQuantizedConv2DOp<quint8, qint8, float, qint8, qint8,
                                             true, false, false, false, false>);
REGISTER_KERNEL_BUILDER(Name("_ZenQuantizedConv2DWithBiasAndRequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint32>("Tbias")
                            .TypeConstraint<qint8>("out_type"),
                        ZenQuantizedConv2DOp<qint8, qint8, qint32, qint8, qint8,
                                             true, false, false, false, false>);
REGISTER_KERNEL_BUILDER(Name("_ZenQuantizedConv2DWithBiasAndRequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<float>("Tbias")
                            .TypeConstraint<qint8>("out_type"),
                        ZenQuantizedConv2DOp<qint8, qint8, float, qint8, qint8,
                                             true, false, false, false, false>);
REGISTER_KERNEL_BUILDER(
    Name("_ZenQuantizedConv2DWithBiasSumAndReluAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<quint8>("out_type")
        .TypeConstraint<qint32>("Tbias"),
    ZenQuantizedConv2DOp<quint8, qint8, qint32, quint8, quint8, true, false,
                         true, true, false>);

REGISTER_KERNEL_BUILDER(
    Name("_ZenQuantizedConv2DWithBiasSumAndReluAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<quint8>("out_type")
        .TypeConstraint<float>("Tbias"),
    ZenQuantizedConv2DOp<quint8, qint8, float, quint8, quint8, true, false,
                         true, true, false>);
REGISTER_KERNEL_BUILDER(
    Name("_ZenQuantizedConv2DWithBiasSignedSumAndReluAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<quint8>("out_type")
        .TypeConstraint<qint32>("Tbias"),
    ZenQuantizedConv2DOp<quint8, qint8, qint32, quint8, qint8, true, false,
                         true, true, true>);

REGISTER_KERNEL_BUILDER(
    Name("_ZenQuantizedConv2DWithBiasSignedSumAndReluAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<quint8>("out_type")
        .TypeConstraint<float>("Tbias"),
    ZenQuantizedConv2DOp<quint8, qint8, float, quint8, qint8, true, false, true,
                         true, true>);

TF_CALL_float(REGISTER_FUSED_CONV2D_KERNELS);
TF_CALL_bfloat16(REGISTER_FUSED_CONV2D_KERNELS);

#undef REGISTER_FUSED_CONV2D_KERNELS

REGISTER_KERNEL_BUILDER(
    Name("_ZenFusedConv2DSum").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    ZenFusedConv2DOp<float, false, false, true>);

}  // namespace amd_cpu_plugin
