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

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_CPU_INFO_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_CPU_INFO_H_
namespace amd_cpu_plugin {
namespace port {

int NumSchedulableCPUs();

int NumHyperthreadsPerCore();

int CPUIDNumSMT();

}  // namespace port
}  // namespace amd_cpu_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_CPU_INFO_H_
