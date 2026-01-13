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
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// TensorFlow plug-in headers.
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/pooling_ops_common.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_kernel_common.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_zendnnl_utils.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_requires.h"
#include "tensorflow_plugin/src/amd_cpu/util/register_types.h"
#include "tensorflow_plugin/src/amd_cpu/util/tensor_format.h"
#include "tensorflow_plugin/src/amd_cpu/util/zen_utils.h"

// ZenDNNL Low Overhead API headers
#include "lowoha_operators/pooling/lowoha_pooling.hpp"
#include "lowoha_operators/pooling/lowoha_pooling_common.hpp"
#include "lowoha_operators/pooling/lowoha_pooling_utils.hpp"

namespace amd_cpu_plugin {

// ZenDNNL Pooling implementation using Low Overhead API (pooling_direct).
template <typename T, bool is_maxpool>
bool TryExecuteZenDNNLPooling(const Tensor &input, Tensor *output,
                              const PoolParameters &params,
                              uint32_t padding_h_top, uint32_t padding_h_bottom,
                              uint32_t padding_w_left,
                              uint32_t padding_w_right) {
  try {
    using namespace zendnnl::lowoha::pooling;
    using namespace zendnnl::common;

    // Get pointers to TensorFlow tensor data.
    T *input_data = const_cast<T *>(input.flat<T>().data());
    T *output_data = output->flat<T>().data();

    // Setup lowoha pooling parameters structure
    pool_params pooling_params;

    // Set pooling dimensions
    pooling_params.dims.batch = params.tensor_in_batch;
    pooling_params.dims.in_height = params.tensor_in_rows;
    pooling_params.dims.in_width = params.tensor_in_cols;
    pooling_params.dims.channels = params.depth;
    pooling_params.dims.kernel_height = params.window_rows;
    pooling_params.dims.kernel_width = params.window_cols;
    pooling_params.dims.out_height = params.out_height;
    pooling_params.dims.out_width = params.out_width;

    // Set pooling parameters
    pooling_params.stride_h = params.row_stride;
    pooling_params.stride_w = params.col_stride;
    pooling_params.pad_top = padding_h_top;
    pooling_params.pad_left = padding_w_left;
    pooling_params.pad_bottom = padding_h_bottom;
    pooling_params.pad_right = padding_w_right;
    pooling_params.is_max_pooling = is_maxpool;
    pooling_params.avg_mode = avg_pooling_mode_t::exclude_padding;
    std::strncpy(pooling_params.data_format, "NHWC", 8);

    // Set data types - with BF16 support enabled
    if (std::is_same<T, float>::value) {
      pooling_params.dtypes.src = data_type_t::f32;
      pooling_params.dtypes.dst = data_type_t::f32;
    } else if (std::is_same<T, Eigen::bfloat16>::value) {
      pooling_params.dtypes.src = data_type_t::bf16;
      pooling_params.dtypes.dst = data_type_t::bf16;
    } else {
      return false;
    }

    // Call ZenDNNL Low Overhead API - pooling_direct with unified params
    status_t status = pooling_direct(input_data, output_data, pooling_params);

    if (status != status_t::success) {
      return false;
    }

    return true;

  } catch (const std::exception &e) {
    return false;
  }
}

// Specialized versions for float and bfloat16
template bool TryExecuteZenDNNLPooling<float, true>(const Tensor &, Tensor *,
                                                    const PoolParameters &,
                                                    uint32_t, uint32_t,
                                                    uint32_t, uint32_t);
template bool TryExecuteZenDNNLPooling<float, false>(const Tensor &, Tensor *,
                                                     const PoolParameters &,
                                                     uint32_t, uint32_t,
                                                     uint32_t, uint32_t);
// BF16 support enabled
template bool TryExecuteZenDNNLPooling<Eigen::bfloat16, true>(
    const Tensor &, Tensor *, const PoolParameters &, uint32_t, uint32_t,
    uint32_t, uint32_t);
template bool TryExecuteZenDNNLPooling<Eigen::bfloat16, false>(
    const Tensor &, Tensor *, const PoolParameters &, uint32_t, uint32_t,
    uint32_t, uint32_t);

template <typename T, bool is_maxpool>
class ZenPoolOp : public OpKernel {
 public:
  explicit ZenPoolOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(
        context, ksize_.size() == 4,
        errors::InvalidArgument("Kernel size field must specify 4 dimensions"));

    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument(
                    "Sliding window stride field must specify 4 dimensions"));

    string padding_str = "";
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_str));
    if (padding_str == "VALID") {
      padding_ = Padding::VALID;
    } else if (padding_str == "SAME") {
      padding_ = Padding::SAME;
    } else {
      padding_ = Padding::EXPLICIT;
    }
    if (context->HasAttr("explicit_paddings")) {
      OP_REQUIRES_OK(
          context, context->GetAttr("explicit_paddings", &explicit_paddings_));
    }

    string data_format_str = "";
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(context, FormatFromString(data_format_str, &data_format_),
                errors::InvalidArgument("Invalid data format"));
  }

  void Compute(OpKernelContext *context) override {
    int data_format = (data_format_ == FORMAT_NCHW) ? 1 : 0;
    const Tensor &input = context->input(0);
    const T *input_array = const_cast<T *>(input.template flat<T>().data());

    PoolParameters params{context,
                          ksize_,
                          stride_,
                          padding_,
                          /*explict padding*/ {},
                          data_format_,
                          input.shape()};
    TensorShape out_shape = params.forward_output_shape();
    bool is_input_float = std::is_same<T, float>::value;

    // Output tensor.
    Tensor *output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    T *output_array = const_cast<T *>(output->template flat<T>().data());

    const int image_height = params.tensor_in_rows;
    const int image_width = params.tensor_in_cols;

    int stride_h, stride_w, filter_height, filter_width;
    int padding_h_top, padding_h_bottom, padding_w_left, padding_w_right;

    stride_h = stride_[1];
    stride_w = stride_[2];

    filter_height = ksize_[1];
    filter_width = ksize_[2];

    // TODO(plugin): Define new function compute_padding for this as it's used
    // at multiple places.
    if (!(padding_ == SAME)) {
      padding_h_top = padding_h_bottom = padding_w_left = padding_w_right = 0;
    } else {
      int total_pad_h, total_pad_w;
      int mod_h, mod_w;
      mod_h = image_height % stride_h;
      mod_w = image_width % stride_w;

      total_pad_h =
          std::max(filter_height - (mod_h == 0 ? stride_h : mod_h), 0);
      padding_h_top =
          (total_pad_h / 2);  // Integer division equivalent to floor.
      padding_h_bottom = total_pad_h - padding_h_top;

      total_pad_w = std::max(filter_width - (mod_w == 0 ? stride_w : mod_w), 0);
      padding_w_left =
          (total_pad_w / 2);  // Integer division equivalent to floor.
      padding_w_right = total_pad_w - padding_w_left;
    }

    bool zendnnl_success = TryExecuteZenDNNLPooling<T, is_maxpool>(
        input, output, params, padding_h_top, padding_h_bottom, padding_w_left,
        padding_w_right);

    OP_REQUIRES(context, zendnnl_success,
                errors::Internal("ZenDNNL Pooling execution failed"));
  }

 private:
  std::vector<int32> ksize_ = {};
  std::vector<int32> stride_ = {};
  Padding padding_ = Padding::VALID;
  std::vector<int64_t> explicit_paddings_ = {};
  // FORMAT_NHWC is the default data format in TensorFlow. Hence initializing
  // with it. Reference from tensorflow_plugin/src/amd_cpu/util/tensor_format.h
  TensorFormat data_format_ = TensorFormat::FORMAT_NHWC;
};

#define REGISTER_POOL_KERNELS(TYPE)                                     \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("_ZenMaxPool").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      ZenPoolOp<TYPE, true>);                                           \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("_ZenAvgPool").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      ZenPoolOp<TYPE, false>);

TF_CALL_float(REGISTER_POOL_KERNELS);
TF_CALL_bfloat16(REGISTER_POOL_KERNELS);

#undef REGISTER_POOL_KERNELS

}  // namespace amd_cpu_plugin
