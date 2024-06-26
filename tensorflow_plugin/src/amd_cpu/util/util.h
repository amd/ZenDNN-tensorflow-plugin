/*******************************************************************************
 * Modifications Copyright (c) 2024 Advanced Micro Devices, Inc. All rights
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

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_UTIL_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_UTIL_H_

#include <string>

#include "tensorflow_plugin/src/amd_cpu/util/tensor_shape.h"

namespace amd_cpu_plugin {

bool IsZenDnnEnabled();
bool IsZenDnnBF16Enabled();
int64_t GetMempool();

std::string SliceDebugString(const TensorShape& shape, const int64 flat);

}  // namespace amd_cpu_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_UTIL_H_
