/*******************************************************************************
 * Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

// TensorFlow C API headers.
#include "tensorflow/c/kernels.h"
// TensorFlow plug-in headers.
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_kernels_init.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_kernel.h"

void RegisterZenKernels() {
  amd_cpu_plugin::register_kernel::RegisterCPUKernels(
      amd_cpu_plugin::DEVICE_CPU);
}
