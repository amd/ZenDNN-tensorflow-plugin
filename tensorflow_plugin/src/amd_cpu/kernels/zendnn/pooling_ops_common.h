/*******************************************************************************
 * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights
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

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_POOLING_OPS_COMMON_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_POOLING_OPS_COMMON_H_

#include <vector>

#include "tensorflow_plugin/src/amd_cpu/util/op_kernel.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_requires.h"
#include "tensorflow_plugin/src/amd_cpu/util/padding.h"
#include "tensorflow_plugin/src/amd_cpu/util/tensor_format.h"

namespace amd_cpu_plugin {

// A helper class to manage sizes and shapes for pooling operations.
struct PoolParameters {
  // Updates context->status if there is an invalid input.
  // explicit_paddings has eight elements if padding==EXPLIICT, and zero
  // elements otherwise.
  PoolParameters(OpKernelContext* context, const std::vector<int32>& ksize,
                 const std::vector<int32>& stride, Padding padding,
                 std::vector<int64_t> explicit_paddings,
                 TensorFormat data_format, const TensorShape& tensor_in_shape);

  // Returns the shape of the output for "forward" pooling operations.
  TensorShape forward_output_shape();

  int depth = 0;

  int tensor_in_cols = 0;
  int tensor_in_rows = 0;
  int tensor_in_batch = 0;

  int window_rows = 0;
  int window_cols = 0;
  int depth_window = 0;

  int row_stride = 0;
  int col_stride = 0;
  int depth_stride = 0;

  int64_t out_height = 0;
  int64_t out_width = 0;
  int out_depth = 0;

  int64_t pad_top = 0;
  int64_t pad_bottom = 0;
  int64_t pad_left = 0;
  int64_t pad_right = 0;

  int pad_depth = 0;

  TensorFormat data_format = TensorFormat::FORMAT_NHWC;
};

}  // namespace amd_cpu_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_POOLING_OPS_COMMON_H_
