/*******************************************************************************
 * Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights
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

// Helper routines for encoding/decoding tensor contents.
#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_TENSOR_CODING_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_TENSOR_CODING_H_

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow_plugin/src/amd_cpu/util/platform.h"
#include "tensorflow_plugin/src/amd_cpu/util/protobuf.h"
#include "tensorflow_plugin/src/amd_cpu/util/refcount.h"
#include "tensorflow_plugin/src/amd_cpu/util/stringpiece.h"
#include "tensorflow_plugin/src/amd_cpu/util/types.h"

namespace amd_cpu_plugin {
namespace port {

using std::string;

// Store src contents in *out.  If backing memory for src is shared with *out,
// will ref obj during the call and will arrange to unref obj when no
// longer needed.
void AssignRefCounted(StringPiece src, core::RefCounted* obj, std::string* out);

// Copy contents of src to dst[0,src.size()-1].
inline void CopyToArray(const std::string& src, char* dst) {
  std::copy_n(src.data(), src.size(), dst);
}

// Copy subrange [pos:(pos + n)) from src to dst. If pos >= src.size() the
// result is empty. If pos + n > src.size() the subrange [pos, size()) is
// copied.
inline void CopySubrangeToArray(const std::string& src, size_t pos, size_t n,
                                char* dst) {
  if (pos >= src.size()) return;
  std::copy_n(src.data() + pos, std::min(n, src.size() - pos), dst);
}

// Store encoding of strings[0..n-1] in *out.
void EncodeStringList(const tstring* strings, int64_t n, std::string* out);

// Decode n strings from src and store in strings[0..n-1].
// Returns true if successful, false on parse error.
bool DecodeStringList(const std::string& src, tstring* strings, int64_t n);

// Assigns base[0..bytes-1] to *s.
void CopyFromArray(std::string* s, const char* base, size_t bytes);

// Encodes sequences of strings and serialized protocol buffers into a string.
// Normal usage consists of zero or more calls to Append() and a single call to
// Finalize().
class StringListEncoder {
 public:
  virtual ~StringListEncoder() = default;

  // Encodes the given protocol buffer. This may not be called after Finalize().
  virtual void Append(const protobuf::MessageLite& m) = 0;

  // Encodes the given string. This may not be called after Finalize().
  virtual void Append(const std::string& s) = 0;

  // Signals end of the encoding process. No other calls are allowed after this.
  virtual void Finalize() = 0;
};

// Decodes a string into sequences of strings (which may represent serialized
// protocol buffers). Normal usage involves a single call to ReadSizes() in
// order to retrieve the length of all the strings in the sequence. For each
// size returned a call to Data() is expected and will return the actual
// string.
class StringListDecoder {
 public:
  virtual ~StringListDecoder() = default;

  // Populates the given vector with the lengths of each string in the sequence
  // being decoded. Upon returning the vector is guaranteed to contain as many
  // elements as there are strings in the sequence.
  virtual bool ReadSizes(std::vector<uint32>* sizes) = 0;

  // Returns a pointer to the next string in the sequence, then prepares for the
  // next call by advancing 'size' characters in the sequence.
  virtual const char* Data(uint32 size) = 0;
};

std::unique_ptr<StringListEncoder> NewStringListEncoder(string* out);
std::unique_ptr<StringListDecoder> NewStringListDecoder(const string& in);

#if defined(TENSORFLOW_PROTOBUF_USES_CORD)
// Store src contents in *out.  If backing memory for src is shared with *out,
// will ref obj during the call and will arrange to unref obj when no
// longer needed.
void AssignRefCounted(StringPiece src, core::RefCounted* obj, absl::Cord* out);

// TODO(kmensah): Macro guard this with a check for Cord support.
inline void CopyToArray(const absl::Cord& src, char* dst) {
  src.CopyToArray(dst);
}

// Copy n bytes of src to dst. If pos >= src.size() the result is empty.
// If pos + n > src.size() the subrange [pos, size()) is copied.
inline void CopySubrangeToArray(const absl::Cord& src, int64_t pos, int64_t n,
                                char* dst) {
  src.Subcord(pos, n).CopyToArray(dst);
}

// Store encoding of strings[0..n-1] in *out.
void EncodeStringList(const tstring* strings, int64_t n, absl::Cord* out);

// Decode n strings from src and store in strings[0..n-1].
// Returns true if successful, false on parse error.
bool DecodeStringList(const absl::Cord& src, std::string* strings, int64_t n);
bool DecodeStringList(const absl::Cord& src, tstring* strings, int64_t n);

// Assigns base[0..bytes-1] to *c.
void CopyFromArray(absl::Cord* c, const char* base, size_t bytes);

std::unique_ptr<StringListEncoder> NewStringListEncoder(absl::Cord* out);
std::unique_ptr<StringListDecoder> NewStringListDecoder(const absl::Cord& in);
#endif  // defined(TENSORFLOW_PROTOBUF_USES_CORD)

}  // namespace port
}  // namespace amd_cpu_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_TENSOR_CODING_H_
