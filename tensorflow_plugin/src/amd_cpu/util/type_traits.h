/*******************************************************************************
 * Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights
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

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_TYPE_TRAITS_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_TYPE_TRAITS_H_

#include <limits>
#include <utility>

#include "tensorflow_plugin/src/amd_cpu/util/numeric_types.h"
#include "tensorflow_plugin/src/amd_cpu/util/types.h"

namespace amd_cpu_plugin {

// Functions to define quantization attribute of types.
struct true_type {
  static constexpr bool value = true;
};
struct false_type {
  static constexpr bool value = false;
};

// Default is_quantized is false.
template <typename T>
struct is_quantized : false_type {};

// Specialize the quantized types.
template <>
struct is_quantized<qint8> : true_type {};
template <>
struct is_quantized<quint8> : true_type {};
template <>
struct is_quantized<qint32> : true_type {};
template <>
struct is_quantized<qint16> : true_type {};
template <>
struct is_quantized<quint16> : true_type {};

// Default is_complex is false.
template <typename T>
struct is_complex : false_type {};

// Specialize std::complex<float> and std::complex<double> types.
template <>
struct is_complex<std::complex<float>> : true_type {};
template <>
struct is_complex<std::complex<double>> : true_type {};

// is_simple_type<T>::value if T[] can be safely constructed and destructed
// without running T() and ~T().  We do not use std::is_trivial<T>
// directly because std::complex<float> and std::complex<double> are
// not trivial, but their arrays can be constructed and destructed
// without running their default ctors and dtors.
template <typename T>
struct is_simple_type {
  static constexpr bool value =
      std::is_trivial<T>::value || std::is_same<T, Eigen::half>::value ||
      std::is_same<T, complex64>::value || std::is_same<T, complex128>::value ||
      is_quantized<T>::value || std::is_same<T, Eigen::bfloat16>::value;
};

}  // namespace amd_cpu_plugin

// Define numeric limits for our quantized as subclasses of the
// standard types.
namespace std {
template <>
class numeric_limits<amd_cpu_plugin::qint8>
    : public numeric_limits<amd_cpu_plugin::int8> {};
template <>
class numeric_limits<amd_cpu_plugin::quint8>
    : public numeric_limits<amd_cpu_plugin::uint8> {};
template <>
class numeric_limits<amd_cpu_plugin::qint16>
    : public numeric_limits<amd_cpu_plugin::int16> {};
template <>
class numeric_limits<amd_cpu_plugin::quint16>
    : public numeric_limits<amd_cpu_plugin::uint16> {};
template <>
class numeric_limits<amd_cpu_plugin::qint32>
    : public numeric_limits<amd_cpu_plugin::int32> {};

// Specialize is_signed for quantized types.
template <>
struct is_signed<amd_cpu_plugin::qint8>
    : public is_signed<amd_cpu_plugin::int8> {};
template <>
struct is_signed<amd_cpu_plugin::quint8>
    : public is_signed<amd_cpu_plugin::uint8> {};
template <>
struct is_signed<amd_cpu_plugin::qint16>
    : public is_signed<amd_cpu_plugin::int16> {};
template <>
struct is_signed<amd_cpu_plugin::quint16>
    : public is_signed<amd_cpu_plugin::uint16> {};
template <>
struct is_signed<amd_cpu_plugin::qint32>
    : public is_signed<amd_cpu_plugin::int32> {};

}  // namespace std

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_TYPE_TRAITS_H_
