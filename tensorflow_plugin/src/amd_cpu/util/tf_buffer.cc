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

#include "tensorflow_plugin/src/amd_cpu/util/tf_buffer.h"

namespace amd_cpu_plugin {

Status MessageToBuffer(const amd_cpu_plugin::protobuf::MessageLite& in,
                       TF_Buffer* out) {
  if (out->data != nullptr) {
    return errors::InvalidArgument("Passing non-empty TF_Buffer is invalid.");
  }
  const size_t proto_size = in.ByteSizeLong();
  void* buf = malloc(proto_size);
  if (buf == nullptr) {
    return errors::ResourceExhausted(
        "Failed to allocate memory to serialize message of type '",
        in.GetTypeName(), "' and size ", proto_size);
  }
  if (!in.SerializeWithCachedSizesToArray(static_cast<uint8*>(buf))) {
    free(buf);
    return errors::InvalidArgument(
        "Unable to serialize ", in.GetTypeName(),
        " protocol buffer, perhaps the serialized size (", proto_size,
        " bytes) is too large?");
  }
  out->data = buf;
  out->length = proto_size;
  out->data_deallocator = [](void* data, size_t length) { free(data); };
  return OkStatus();
}

Status BufferToMessage(
    const TF_Buffer* in,
    amd_cpu_plugin::protobuf::MessageLite& out) {  // NOLINT(runtime/references)
  if (in == nullptr || !out.ParseFromArray(in->data, in->length)) {
    return errors::InvalidArgument("Unparseable proto");
  }
  return OkStatus();
}

}  // namespace amd_cpu_plugin
