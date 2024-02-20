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

#include "tensorflow_plugin/src/amd_cpu/util/util.h"

#include "absl/base/call_once.h"
#include "tensorflow_plugin/src/amd_cpu/util/gtl/inlined_vector.h"
#include "tensorflow_plugin/src/amd_cpu/util/strcat.h"
#include "tensorflow_plugin/src/amd_cpu/util/types.h"
#include "tensorflow_plugin/src/amd_cpu/util/zen_utils.h"

namespace amd_cpu_plugin {

bool IsZenDnnEnabled() {
  static absl::once_flag once;
  static bool ZenDNN_enabled = false;
  absl::call_once(once, [&] {
    auto status = ReadBoolFromEnvVar("TF_ENABLE_ZENDNN_OPTS", ZenDNN_enabled,
                                     &ZenDNN_enabled);

    if (!status.ok()) {
      zendnnInfo(ZENDNN_FWKLOG,
                 "TF_ENABLE_ZENDNN_OPTS is not set to either '0', 'false', "
                 "'1', or 'true'. Using the default setting: ",
                 ZenDNN_enabled);
    }
  });
  return ZenDNN_enabled;
}

bool IsZenDnnBF16Enabled() {
  static absl::once_flag once;
  static bool tf_zendnn_plugin_bf16 = false;
  absl::call_once(once, [&] {
    auto status = ReadBoolFromEnvVar(
        "TF_ZENDNN_PLUGIN_BF16", tf_zendnn_plugin_bf16, &tf_zendnn_plugin_bf16);
    if (tf_zendnn_plugin_bf16) {
      // Check for the BF16 support on the machine.
      bool result = tensorflow::port::TestCPUFeature(
          tensorflow::port::CPUFeature::AVX512F);
      if (!result) {
        LOG(INFO)
            << " BF16 AVX512 instruction set is not supported in the machine."
            << " Auto_Mixed_Precision can't be enabled."
            << " Hence, default FP32 precision type is used.";
        tf_zendnn_plugin_bf16 = false;
      }
    }
    if (!status.ok()) {
      zendnnInfo(ZENDNN_FWKLOG,
                 "TF_ZENDNN_PLUGIN_BF16 is not set to either '0', 'false', "
                 "or '1', 'true'. Using the default setting: ",
                 tf_zendnn_plugin_bf16);
    }
  });
  return tf_zendnn_plugin_bf16;
}

int64_t GetMempool() {
  static absl::once_flag once;
  static int64_t mempool = 1;
  absl::call_once(once, [&] {
    auto status =
        ReadInt64FromEnvVar("ZENDNN_ENABLE_MEMPOOL", mempool, &mempool);

    if (!status.ok()) {
      zendnnInfo(
          ZENDNN_FWKLOG,
          "ZENDNN_ENABLE_MEMPOOL is not set. Using the default setting: ",
          mempool);
    }
  });
  return mempool;
}

std::string SliceDebugString(const TensorShape& shape, const int64 flat) {
  // Special case rank 0 and 1
  const int dims = shape.dims();
  if (dims == 0) return "";
  if (dims == 1) return strings::StrCat("[", flat, "]");

  // Compute strides
  gtl::InlinedVector<int64, 32> strides(dims);
  strides.back() = 1;
  for (int i = dims - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * shape.dim_size(i + 1);
  }

  // Unflatten index
  int64 left = flat;
  string result;
  for (int i = 0; i < dims; i++) {
    strings::StrAppend(&result, i ? "," : "[", left / strides[i]);
    left %= strides[i];
  }
  strings::StrAppend(&result, "]");
  return result;
}

}  // namespace amd_cpu_plugin
