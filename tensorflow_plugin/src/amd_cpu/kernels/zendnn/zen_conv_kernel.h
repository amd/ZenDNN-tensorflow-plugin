/*******************************************************************************
 * Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 *******************************************************************************/

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

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_CONV_KERNEL_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_CONV_KERNEL_H_

// Standard headers
#include <limits>
#include <memory>
#include <string>
#include <vector>
// TensorFlow plug-in headers
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/fused_eigen_output_kernels.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_conv_kernel_util.h"
#include "tensorflow_plugin/src/amd_cpu/util/bounds_check.h"
#include "tensorflow_plugin/src/amd_cpu/util/common_shape_fns.h"
#include "tensorflow_plugin/src/amd_cpu/util/errors.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_kernel.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_requires.h"
#include "tensorflow_plugin/src/amd_cpu/util/padding.h"
#include "tensorflow_plugin/src/amd_cpu/util/tensor_format.h"
#include "tensorflow_plugin/src/amd_cpu/util/zen_utils.h"

namespace amd_cpu_plugin {

// Convolution parameters specified by Op attributes.
struct Conv2DParameters {
  std::vector<int32> dilations;
  std::vector<int32> strides;
  Padding padding;
  TensorFormat data_format;
  std::vector<int64_t> explicit_paddings;
};

// Convolution dimensions inferred from parameters, input and filter tensors.
struct Conv2DDimensions {
  int batch;
  int input_rows;
  int input_cols;
  int in_depth;

  int filter_rows;
  int filter_cols;
  int patch_depth;
  int out_depth;

  int stride_rows;
  int stride_cols;

  int dilation_rows;
  int dilation_cols;

  int64_t out_rows;
  int64_t out_cols;
  int64_t pad_rows_before;
  int64_t pad_rows_after;
  int64_t pad_cols_before;
  int64_t pad_cols_after;
};

Status InitConv2DParameters(const OpKernelConstruction* context,
                            Conv2DParameters* params);

class ConvUtil {
 public:
  ConvUtil(OpKernelContext* context, const Conv2DParameters& params,
           bool is_depthwise)
      : context_(context),
        data_format_(params.data_format),
        strides_(params.strides),
        dilations_(params.dilations),
        padding_(params.padding),
        explicit_paddings_(params.explicit_paddings),
        is_depthwise_(is_depthwise) {}

  virtual ~ConvUtil() { context_ = nullptr; }

  // Calculate Convolution strides
  virtual inline void GetStrideDimension(Conv2DDimensions* dimensions) {
    // For now we take the stride from the second and third dimensions only
    // (we do not support striding on the batch or depth dimension).

    if (strides_.size() == 4) {
      int stride_rows = GetTensorDim(strides_, data_format_, 'H');
      int stride_cols = GetTensorDim(strides_, data_format_, 'W');
      dimensions->stride_rows = stride_rows;
      dimensions->stride_cols = stride_cols;
    } else if (strides_.size() == 5) {
      zendnnInfo(ZENDNN_FWKLOG, "ZEN-OP-DEF: ZenConv3D Error!!");
    }
  }

  // Calculate Convolution dilations
  virtual inline void GetDilationDimension(Conv2DDimensions* dimensions) {
    // For now we take the dilation from the second and third dimensions only
    // (we do not support dilation on the batch or depth dimension).

    if (dilations_.size() == 4) {
      int dilations_rows = GetTensorDim(dilations_, data_format_, 'H');
      int dilations_cols = GetTensorDim(dilations_, data_format_, 'W');
      dimensions->dilation_rows = dilations_rows;
      dimensions->dilation_cols = dilations_cols;
    } else if (dilations_.size() == 5) {
      zendnnInfo(ZENDNN_FWKLOG, "ZEN-OP-DEF: ZenConv3D Error!!");
    }
  }

  // requires input in NCHW/NCDHW format. Function does not return anything.
  // But errors arising from sanity checks are returned in context's
  // status.
  virtual inline void GetInputDimension(const TensorShape& input_shape,
                                        Conv2DDimensions* dimensions) {
#define CHECK_BOUNDS(val, err_msg)                                     \
  do {                                                                 \
    OP_REQUIRES(context_,                                              \
                FastBoundsCheck(val, std::numeric_limits<int>::max()), \
                errors::InvalidArgument(err_msg));                     \
  } while (0)

    // Input channel
    int64 input_depth_raw = GetTensorDim(input_shape, data_format_, 'C');
    int input_depth = static_cast<int>(input_depth_raw);

    // Input batch
    int64 input_batch_raw = GetTensorDim(input_shape, data_format_, 'N');
    CHECK_BOUNDS(input_batch_raw, "Input batch too large");
    int input_batch = static_cast<int>(input_batch_raw);

    if (strides_.size() == 4) {  // NCHW format for Conv2D
      // Input rows/height
      int64 input_rows_raw = GetTensorDim(input_shape, data_format_, 'H');
      CHECK_BOUNDS(input_rows_raw, "Input rows too large");
      int input_rows = static_cast<int>(input_rows_raw);

      // Input columns/width
      int64 input_cols_raw = GetTensorDim(input_shape, data_format_, 'W');
      CHECK_BOUNDS(input_cols_raw, "Input cols too large");
      int input_cols = static_cast<int>(input_cols_raw);

      // ZenDNN always requires input in NHWC format Conv2D.
      dimensions->batch = input_batch;
      dimensions->in_depth = input_depth;
      dimensions->input_rows = input_rows;
      dimensions->input_cols = input_cols;

    } else if (strides_.size() == 5) {  // NCDHW format for Conv3D
      zendnnInfo(ZENDNN_FWKLOG, "ZEN-OP-DEF: ZenConv3D Error!!");
    }
#undef CHECK_BOUNDS
  }

  // Function does not return anything.
  // But errors arising from sanity checks are returned in context's
  // status.
  virtual inline void GetFilterDimension(const TensorShape& input_shape,
                                         const TensorShape& filter_shape,
                                         Conv2DDimensions* dimensions) {
    OP_REQUIRES(context_, filter_shape.dims() == strides_.size(),
                errors::InvalidArgument((strides_.size() == 4)
                                            ? "filter must be 4-dimensional: "
                                            : "filter must be 5-dimensional: ",
                                        filter_shape.DebugString()));

    for (int i = 0; i < ((strides_.size() == 4) ? 3 : 5); i++) {
      OP_REQUIRES(context_,
                  FastBoundsCheck(filter_shape.dim_size(i),
                                  std::numeric_limits<int>::max()),
                  errors::InvalidArgument("filter too large"));
    }

    int input_depth = GetTensorDim(input_shape, data_format_, 'C');

    if (strides_.size() == 4) {  // Conv2D
      OP_REQUIRES(context_, input_depth == filter_shape.dim_size(2),
                  errors::InvalidArgument(
                      "input and filter must have the same depth: ",
                      input_depth, " vs ", filter_shape.dim_size(2)));

      // TF filter is always in (rows, cols, in_depth, out_depth) order.
      // TODO(aakar): These magic numbers can be replaced with enum from
      // zen_utils.h in TF Plugin's util folder
      int filter_rows = static_cast<int>(filter_shape.dim_size(0));
      int filter_cols = static_cast<int>(filter_shape.dim_size(1));
      int filter_in_depth = static_cast<int>(filter_shape.dim_size(2));
      int filter_out_depth = static_cast<int>(filter_shape.dim_size(3));
      // TODO(aakar): Find out ZenDNN convention for depthwise
      // oneDNN always needs filter in OIHW format for regular convolutions
      // and GOIHW for grouped/depthwise convolutions,
      // OIHW = (out_depth, in_depth, rows, cols)
      // GOIHW = (group, out_depth, in_depth, rows, cols)
      // Specifically for depthwise G=filter_indepth, O=filter_outdepth, I=1
      if (is_depthwise_) {
        dimensions->out_depth = filter_out_depth * filter_in_depth;
        dimensions->patch_depth = filter_in_depth;
        dimensions->filter_rows = filter_rows;
        dimensions->filter_cols = filter_cols;
      } else {
        dimensions->out_depth = filter_out_depth;
        dimensions->patch_depth = filter_in_depth;
        dimensions->filter_rows = filter_rows;
        dimensions->filter_cols = filter_cols;
      }
    } else {  // Conv3D
      zendnnInfo(ZENDNN_FWKLOG, "ZEN-OP-DEF: ZenConv3D Error!!");
    }
  }

  // Function to calculate output and padding size for 2D/3D convolution.
  //
  // Calculate output shape of Convolution in ZenDNN and TensorFlow order.
  // Both ZenDNN and TensorFlow output will be in NHWC||NCHW(Conv2D) or
  // NDHWC||NCDHW(Conv3D) format depending on data format.
  // Function also calculates left, right, top and bottom pads.
  // Function does not return any status which is set with context status.
  virtual inline void GetOutputAndPadDimension(const TensorShape& input_shape,
                                               const TensorShape& filter_shape,
                                               Conv2DDimensions* dimensions,
                                               bool pad_enabled = false) {
    bool is_conv2d = (strides_.size() == 4);

    int64 out_rows = 0, out_cols = 0;
    int64 pad_top = 0, pad_bottom = 0, pad_left = 0, pad_right = 0;
    if (is_conv2d) {
      Padding padding_type;
      if (pad_enabled) {  // false by choice
        padding_type = Padding::EXPLICIT;
        zendnnInfo(ZENDNN_FWKLOG, "ZEN-OP-DEF: ZenConv Fuse Error!!");
      } else {
        padding_type = padding_;
        if (padding_type == Padding::EXPLICIT) {
          GetExplicitPaddingForDim(explicit_paddings_, data_format_, 'H',
                                   &pad_top, &pad_bottom);
          GetExplicitPaddingForDim(explicit_paddings_, data_format_, 'W',
                                   &pad_left, &pad_right);
        }
      }

      OP_REQUIRES_OK(context_,
                     GetWindowedOutputSizeVerboseV2(
                         dimensions->input_rows, dimensions->filter_rows,
                         dimensions->dilation_rows, dimensions->stride_rows,
                         padding_type, &out_rows, &pad_top, &pad_bottom));
      OP_REQUIRES_OK(context_,
                     GetWindowedOutputSizeVerboseV2(
                         dimensions->input_cols, dimensions->filter_cols,
                         dimensions->dilation_cols, dimensions->stride_cols,
                         padding_type, &out_cols, &pad_left, &pad_right));
    } else {
      zendnnInfo(ZENDNN_FWKLOG, "ZEN-OP-DEF: ZenConv3D Error!!");
    }

    dimensions->out_rows = out_rows;
    dimensions->out_cols = out_cols;
    dimensions->pad_rows_before = pad_top;
    dimensions->pad_rows_after = pad_bottom;
    dimensions->pad_cols_before = pad_left;
    dimensions->pad_cols_after = pad_right;
  }

  // Wrapper function to calculate input, filter, and output sizes of
  // Additionally, it also calculates strides and paddings.
  //
  // Function does not return anything, but sets error in context status.
  inline void InitFwdDimensions(const TensorShape& input_shape,
                                const TensorShape& filter_shape,
                                Conv2DDimensions* dimensions,
                                bool pad_enabled = false) {
    GetInputDimension(input_shape, dimensions);
    GetFilterDimension(input_shape, filter_shape, dimensions);
    GetStrideDimension(dimensions);
    GetDilationDimension(dimensions);
    GetOutputAndPadDimension(input_shape, filter_shape, dimensions);
  }

 protected:
  OpKernelContext* context_;  // We don't own this.
  TensorFormat data_format_;
  std::vector<int32_t> strides_;
  std::vector<int32_t> dilations_;
  Padding padding_;
  std::vector<int64_t> explicit_paddings_;
  bool is_depthwise_;
};

}  // namespace amd_cpu_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_CONV_KERNEL_H_
