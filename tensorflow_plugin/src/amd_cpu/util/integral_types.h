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

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_INTEGRAL_TYPES_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_INTEGRAL_TYPES_H_

#include <cstdint>

namespace amd_cpu_plugin {

typedef signed char int8;
typedef short int16;  // NOLINT(runtime/int)
typedef int int32;

// for compatible with int64_t
typedef std::int64_t int64;

typedef unsigned char uint8;
typedef unsigned short uint16;  // NOLINT(runtime/int)
typedef unsigned int uint32;
typedef std::uint64_t uint64;

}  // namespace amd_cpu_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_INTEGRAL_TYPES_H_
