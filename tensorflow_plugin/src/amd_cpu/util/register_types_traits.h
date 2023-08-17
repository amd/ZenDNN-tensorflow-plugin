/*******************************************************************************
 * Modifications Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights
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

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_REGISTER_TYPES_TRAITS_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_REGISTER_TYPES_TRAITS_H_

#include "tensorflow_plugin/src/amd_cpu/util/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
typedef Eigen::ThreadPoolDevice CPUDevice;

namespace amd_cpu_plugin {

// Remap POD types by size to equivalent proxy types. This works
// since all we are doing is copying data around.
struct UnusableProxyType;
template <typename Device, int size>
struct proxy_type_pod {
  typedef UnusableProxyType type;
};
template <>
struct proxy_type_pod<CPUDevice, 16> {
  typedef ::amd_cpu_plugin::complex128 type;
};
template <>
struct proxy_type_pod<CPUDevice, 8> {
  typedef ::int64_t type;
};
template <>
struct proxy_type_pod<CPUDevice, 4> {
  typedef ::amd_cpu_plugin::int32 type;
};
template <>
struct proxy_type_pod<CPUDevice, 2> {
  typedef ::amd_cpu_plugin::int16 type;
};
template <>
struct proxy_type_pod<CPUDevice, 1> {
  typedef ::amd_cpu_plugin::int8 type;
};

/// If POD we use proxy_type_pod, otherwise this maps to identity.
template <typename Device, typename T>
struct proxy_type {
  typedef typename std::conditional<
      std::is_arithmetic<T>::value,
      typename proxy_type_pod<Device, sizeof(T)>::type, T>::type type;
  static_assert(sizeof(type) == sizeof(T), "proxy_type_pod is not valid");
};

/// The active proxy types
#define TF_CALL_CPU_PROXY_TYPES(m)                                     \
  TF_CALL_int64(m) TF_CALL_int32(m) TF_CALL_uint16(m) TF_CALL_int16(m) \
      TF_CALL_int8(m) TF_CALL_complex128(m)
}  // namespace amd_cpu_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_REGISTER_TYPES_TRAITS_H_
