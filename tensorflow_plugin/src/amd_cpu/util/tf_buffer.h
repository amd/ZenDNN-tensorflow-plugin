/*******************************************************************************
 * Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 ******************************************************************************/

/* Copyright (c) 2021-2022 Intel Corporation

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

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_TF_BUFFER_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_TF_BUFFER_H_

#include "tensorflow/c/c_api.h"
#include "tensorflow_plugin/src/amd_cpu/util/errors.h"
#include "tensorflow_plugin/src/amd_cpu/util/status.h"

// Import whatever namespace protobuf comes from into the
// ::amd_cpu_plugin::protobuf namespace.
//
// amd_cpu_plugin code should use the ::amd_cpu_plugin::protobuf namespace to
// refer to all protobuf APIs.

#include "google/protobuf/arena.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.pb.h"
#include "google/protobuf/dynamic_message.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/tokenizer.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/map.h"
#include "google/protobuf/message.h"
#include "google/protobuf/repeated_field.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/json_util.h"
#include "google/protobuf/util/type_resolver_util.h"

namespace amd_cpu_plugin {

namespace protobuf = ::google::protobuf;

Status MessageToBuffer(const amd_cpu_plugin::protobuf::MessageLite& in,
                       TF_Buffer* out);

Status BufferToMessage(
    const TF_Buffer* in,
    amd_cpu_plugin::protobuf::MessageLite& out);  // NOLINT(runtime/references)
}  // namespace amd_cpu_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_TF_BUFFER_H_
