/*******************************************************************************
 * Modifications Copyright (c) 2024 Advanced Micro Devices, Inc. All rights
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

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_ENV_VAR_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_ENV_VAR_H_

#include <string>

#include "tensorflow_plugin/src/amd_cpu/util/status.h"

namespace amd_cpu_plugin {

// Returns a boolean into "value" from the environmental variable
// "env_var_name". If it is unset, the default value is used. A string "0" or a
// case insensitive "false" is interpreted as false. A string "1" or a case
// insensitive "true" is interpreted as true. Otherwise, an error status is
// returned.
Status ReadBoolFromEnvVar(StringPiece env_var_name, bool default_val,
                          bool* value);

// Returns a string into "value" from the environmental variable "env_var_name".
// If it is unset, the default value is used.
Status ReadStringFromEnvVar(StringPiece env_var_name, StringPiece default_val,
                            std::string* value);

// Returns an int64 into "value" from the environmental variable "env_var_name".
// If it is unset, the default value is used.
// If the string cannot be parsed into int64, an error status is returned.
Status ReadInt64FromEnvVar(StringPiece env_var_name, int64_t default_val,
                           int64* value);
}  // namespace amd_cpu_plugin

#endif  //  TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_ENV_VAR_H_
