/*******************************************************************************
 * Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights
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

#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_conv_kernel.h"

namespace amd_cpu_plugin {

#define TF_REQUIRES(EXP, STATUS)                \
  do {                                          \
    if (!TF_PREDICT_TRUE(EXP)) return (STATUS); \
  } while (false)

Status InitConv2DParameters(const OpKernelConstruction* context,
                            Conv2DParameters* params) {
  TF_RETURN_IF_ERROR(context->GetAttr("dilations", &params->dilations));
  TF_RETURN_IF_ERROR(context->GetAttr("strides", &params->strides));
  string padding_str;
  TF_RETURN_IF_ERROR(context->GetAttr("padding", &padding_str));
  if (padding_str == "VALID") {
    params->padding = Padding::VALID;
  } else if (padding_str == "SAME") {
    params->padding = Padding::SAME;
  } else if (padding_str == "EXPLICIT") {
    params->padding = Padding::EXPLICIT;
  } else {
    TF_REQUIRES(false, errors::InvalidArgument("Unknown padding type: ",
                                               params->padding));
  }
  if (context->HasAttr("explicit_paddings")) {
    TF_RETURN_IF_ERROR(
        context->GetAttr("explicit_paddings", &params->explicit_paddings));
  }
  string data_format_string;
  TF_RETURN_IF_ERROR(context->GetAttr("data_format", &data_format_string));
  TF_REQUIRES(FormatFromString(data_format_string, &params->data_format),
              errors::InvalidArgument("Invalid data format"));

  const auto& strides = params->strides;
  const auto& dilations = params->dilations;
  const auto& data_format = params->data_format;

  TF_REQUIRES(dilations.size() == 4,
              errors::InvalidArgument("Sliding window dilations field must "
                                      "specify 4 dimensions"));
  TF_REQUIRES(strides.size() == 4,
              errors::InvalidArgument("Sliding window strides field must "
                                      "specify 4 dimensions"));
  const int64_t stride_n = GetTensorDim(strides, data_format, 'N');
  const int64_t stride_c = GetTensorDim(strides, data_format, 'C');
  const int64_t stride_h = GetTensorDim(strides, data_format, 'H');
  const int64_t stride_w = GetTensorDim(strides, data_format, 'W');
  TF_REQUIRES(
      stride_n == 1 && stride_c == 1,
      errors::Unimplemented("Current implementation does not yet support "
                            "strides in the batch and depth dimensions."));
  TF_REQUIRES(stride_h > 0 && stride_w > 0,
              errors::InvalidArgument(
                  "Row and column strides should be larger than 0."));

  const int64_t dilation_n = GetTensorDim(dilations, data_format, 'N');
  const int64_t dilation_c = GetTensorDim(dilations, data_format, 'C');
  const int64_t dilation_h = GetTensorDim(dilations, data_format, 'H');
  const int64_t dilation_w = GetTensorDim(dilations, data_format, 'W');
  TF_REQUIRES(
      dilation_n == 1 && dilation_c == 1,
      errors::Unimplemented("Current implementation does not yet support "
                            "dilations in the batch and depth dimensions."));
  TF_REQUIRES(
      dilation_h > 0 && dilation_w > 0,
      errors::InvalidArgument("Dilated rates should be larger than 0."));

  TF_RETURN_IF_ERROR(CheckValidPadding(params->padding,
                                       params->explicit_paddings,
                                       /*num_dims=*/4, data_format));

  return OkStatus();
}

#undef TF_REQUIRES

}  // namespace amd_cpu_plugin
