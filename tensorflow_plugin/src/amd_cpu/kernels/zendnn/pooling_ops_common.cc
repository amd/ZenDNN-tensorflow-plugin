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

#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/pooling_ops_common.h"

#include <vector>

#include "tensorflow_plugin/src/amd_cpu/util/bounds_check.h"
#include "tensorflow_plugin/src/amd_cpu/util/common_shape_fns.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_requires.h"
#include "tensorflow_plugin/src/amd_cpu/util/types.h"

namespace amd_cpu_plugin {

Status CheckPaddingSize(int64_t window_rows, int64_t window_cols,
                        int64_t pad_top, int64_t pad_bottom, int64_t pad_left,
                        int64_t pad_right) {
  if (!FastBoundsCheck(pad_top, window_rows)) {
    return errors::InvalidArgument("Top padding ", pad_top,
                                   " needs to be smaller than the "
                                   "window size ",
                                   window_rows);
  }
  if (!FastBoundsCheck(pad_bottom, window_rows)) {
    return errors::InvalidArgument("Bottom padding ", pad_bottom,
                                   " needs to be smaller than the "
                                   "window size ",
                                   window_rows);
  }
  if (!FastBoundsCheck(pad_left, window_cols)) {
    return errors::InvalidArgument("Left padding ", pad_left,
                                   " needs to be smaller than the "
                                   "window size ",
                                   window_cols);
  }
  if (!FastBoundsCheck(pad_right, window_cols)) {
    return errors::InvalidArgument("Right padding ", pad_right,
                                   " needs to be smaller than the "
                                   "window size ",
                                   window_cols);
  }
  return OkStatus();
}

PoolParameters::PoolParameters(OpKernelContext* context,
                               const std::vector<int32>& ksize,
                               const std::vector<int32>& stride,
                               Padding padding,
                               std::vector<int64_t> explicit_paddings,
                               TensorFormat data_format,
                               const TensorShape& tensor_in_shape) {
  // For maxpooling, tensor_in should have 2 spatial dimensions.
  // Note: The total number of dimensions could be 4 for NHWC, NCHW,
  // or 5 for NCHW_VECT_C.
  OP_REQUIRES(context,
              GetTensorSpatialDims(tensor_in_shape.dims(), data_format) == 2,
              errors::InvalidArgument(
                  "tensor_in_shape must have 2 spatial dimensions. ",
                  tensor_in_shape.dims(), " ", data_format));

  this->data_format = data_format;
  depth = GetTensorDim(tensor_in_shape, data_format, 'C') *
          (data_format == FORMAT_NCHW_VECT_C ? 4 : 1);
  tensor_in_cols = GetTensorDim(tensor_in_shape, data_format, 'W');
  tensor_in_rows = GetTensorDim(tensor_in_shape, data_format, 'H');
  tensor_in_batch = GetTensorDim(tensor_in_shape, data_format, 'N');
  window_rows = GetTensorDim(ksize, data_format, 'H');
  window_cols = GetTensorDim(ksize, data_format, 'W');
  depth_window = GetTensorDim(ksize, data_format, 'C');
  row_stride = GetTensorDim(stride, data_format, 'H');
  col_stride = GetTensorDim(stride, data_format, 'W');
  depth_stride = GetTensorDim(stride, data_format, 'C');

  // We only support 2D pooling across width/height and depthwise
  // pooling, not a combination.
  OP_REQUIRES(context,
              (depth_window == 1 || (window_rows == 1 && window_cols == 1)),
              errors::Unimplemented(
                  "MaxPooling supports exactly one of pooling across depth "
                  "or pooling across width/height."));
  if (padding == Padding::EXPLICIT) {
    OP_REQUIRES_OK(context, CheckValidPadding(padding, explicit_paddings,
                                              /*num_dims=*/4, data_format));
    GetExplicitPaddingForDim(explicit_paddings, data_format, 'H', &pad_top,
                             &pad_bottom);
    GetExplicitPaddingForDim(explicit_paddings, data_format, 'W', &pad_left,
                             &pad_right);
    OP_REQUIRES_OK(context, CheckPaddingSize(window_rows, window_cols, pad_top,
                                             pad_bottom, pad_left, pad_right));
  }

  if (depth_window == 1) {
    OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                tensor_in_rows, window_rows, row_stride,
                                padding, &out_height, &pad_top, &pad_bottom));
    OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                tensor_in_cols, window_cols, col_stride,
                                padding, &out_width, &pad_left, &pad_right));
    pad_depth = 0;
    out_depth = depth;
  } else {
    OP_REQUIRES(context, depth_window > 0,
                errors::InvalidArgument("depth_window must not be 0"));
    // Our current version of depthwise max pooling does not support
    // any padding, and expects the depth_window to equal the
    // depth_stride (no overlapping).
    OP_REQUIRES(
        context, depth % depth_window == 0,
        errors::Unimplemented("Depthwise max pooling requires the depth "
                              "window to evenly divide the input depth"));
    OP_REQUIRES(
        context, depth_stride == depth_window,
        errors::Unimplemented("Depthwise max pooling requires the depth "
                              "window to equal the depth stride"));

    pad_depth = 0;
    out_depth = depth / depth_window;
  }
}

TensorShape PoolParameters::forward_output_shape() {
  if (depth_window == 1) {
    // Spatial pooling.
    return ShapeFromFormat(data_format, tensor_in_batch, out_height, out_width,
                           depth);
  } else {
    // Depthwise pooling.
    return TensorShape(
        {tensor_in_batch, tensor_in_rows, tensor_in_cols, out_depth});
  }
}

}  // namespace amd_cpu_plugin
