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

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_PADDING_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_PADDING_H_

#include <string>
#include <vector>

#include "tensorflow_plugin/src/amd_cpu/util/status.h"
#include "tensorflow_plugin/src/amd_cpu/util/tensor_format.h"

namespace amd_cpu_plugin {
// Padding: the padding we apply to the input tensor along the rows and columns
// dimensions. This is usually used to make sure that the spatial dimensions do
// not shrink when we progress with convolutions. Three types of padding are
// supported:
//   VALID: No padding is carried out.
//   SAME: The pad value is computed so that the output will have the same
//         dimensions as the input.
//   EXPLICIT: The user specifies the pad values in the explicit_paddings
//             attribute.
// The padded area is typically zero-filled. For pooling ops, the padded area is
// instead ignored. For max pool, this is equivalent to padding with -infinity.
enum Padding {
  VALID = 1,     // No padding.
  SAME = 2,      // Input and output layers have the same size.
  EXPLICIT = 3,  // Padding is explicitly specified
};

// Returns an error if the padding attributes are invalid.
Status CheckValidPadding(Padding padding_type,
                         const std::vector<int64_t>& explicit_paddings,
                         int num_dims, TensorFormat data_format);

// Return the string containing the list of valid padding types, that can be
// used as an Attr() in REGISTER_OP.
std::string GetPaddingAttrString();

// Like GetPaddingAttrString(), but also includes EXPLICIT.
std::string GetPaddingAttrStringWithExplicit();

std::string GetExplicitPaddingsAttrString();

// Sets padding value based on the given string padding value.
Status GetPaddingFromString(StringPiece str_value, Padding* value);

}  // namespace amd_cpu_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_PADDING_H_
