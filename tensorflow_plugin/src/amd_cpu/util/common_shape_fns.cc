/*******************************************************************************
 * Modifications Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights
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

#include "tensorflow_plugin/src/amd_cpu/util/common_shape_fns.h"

#include <algorithm>

#include "tensorflow_plugin/src/amd_cpu/util/errors.h"
#include "tensorflow_plugin/src/amd_cpu/util/types.h"

namespace amd_cpu_plugin {

Status GetWindowedOutputSizeVerboseV2(int64 input_size, int64 filter_size,
                                      int64 dilation_rate, int64 stride,
                                      Padding padding_type, int64* output_size,
                                      int64* padding_before,
                                      int64* padding_after) {
  if (stride <= 0) {
    return errors::InvalidArgument("Stride must be > 0, but got ", stride);
  }
  if (dilation_rate < 1) {
    return errors::InvalidArgument("Dilation rate must be >= 1, but got ",
                                   dilation_rate);
  }

  // See also the parallel implementation in GetWindowedOutputSizeFromDimsV2.
  int64 effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  switch (padding_type) {
    case Padding::VALID:
      *output_size = (input_size - effective_filter_size + stride) / stride;
      *padding_before = *padding_after = 0;
      break;
    case Padding::EXPLICIT:
      *output_size = (input_size + *padding_before + *padding_after -
                      effective_filter_size + stride) /
                     stride;
      break;
    case Padding::SAME:
      *output_size = (input_size + stride - 1) / stride;
      const int64 padding_needed =
          std::max(int64{0}, (*output_size - 1) * stride +
                                 effective_filter_size - input_size);
      // For odd values of total padding, add more padding at the 'right'
      // side of the given dimension.
      *padding_before = padding_needed / 2;
      *padding_after = padding_needed - *padding_before;
      break;
  }
  if (*output_size < 0) {
    return errors::InvalidArgument(
        "Computed output size would be negative: ", *output_size,
        " [input_size: ", input_size,
        ", effective_filter_size: ", effective_filter_size,
        ", stride: ", stride, "]");
  }
  return OkStatus();
}

Status GetWindowedOutputSizeVerbose(int64 input_size, int64 filter_size,
                                    int64 stride, Padding padding_type,
                                    int64* output_size, int64* padding_before,
                                    int64* padding_after) {
  return GetWindowedOutputSizeVerboseV2(input_size, filter_size,
                                        /*dilation_rate=*/1, stride,
                                        padding_type, output_size,
                                        padding_before, padding_after);
}

}  // namespace amd_cpu_plugin
