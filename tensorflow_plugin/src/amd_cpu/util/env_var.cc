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

#include "tensorflow_plugin/src/amd_cpu/util/env_var.h"

#include <stdlib.h>

#include <string>

#include "tensorflow_plugin/src/amd_cpu/util/errors.h"
#include "tensorflow_plugin/src/amd_cpu/util/strcat.h"

namespace amd_cpu_plugin {

Status ReadBoolFromEnvVar(StringPiece env_var_name, bool default_val,
                          bool* value) {
  *value = default_val;
  const char* tf_env_var_val = getenv(string(env_var_name).c_str());
  if (tf_env_var_val == nullptr) {
    return OkStatus();
  }
  string str_value = absl::AsciiStrToLower(tf_env_var_val);
  if (str_value == "0" || str_value == "false") {
    *value = false;
    return OkStatus();
  } else if (str_value == "1" || str_value == "true") {
    *value = true;
    return OkStatus();
  }
  return errors::InvalidArgument(strings::StrCat(
      "Failed to parse the env-var ${", env_var_name, "} into bool: ",
      tf_env_var_val, ". Use the default value: ", default_val));
}

}  // namespace amd_cpu_plugin
