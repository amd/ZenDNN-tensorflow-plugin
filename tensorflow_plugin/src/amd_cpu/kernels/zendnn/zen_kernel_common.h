/*******************************************************************************
 * Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 ******************************************************************************/

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_KERNEL_COMMON_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_KERNEL_COMMON_H_

#include "tensorflow_plugin/src/amd_cpu/util/op_kernel.h"

namespace amd_cpu_plugin {

// Tensor type enumeration for ZenDNN operations.
enum class ZenTensorType { kQint8 = 0, kQuint8 = 1, kFloat = 2, kBfloat16 = 3 };

}  // namespace amd_cpu_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_KERNEL_COMMON_H_
