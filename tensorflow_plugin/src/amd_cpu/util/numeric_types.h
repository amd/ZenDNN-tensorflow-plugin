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

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_NUMERIC_TYPES_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_NUMERIC_TYPES_H_

#include <complex>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
// Disable clang-format to prevent 'FixedPoint' header from being included
// before 'Tensor' header on which it depends.
// clang-format off
#include "third_party/eigen3/unsupported/Eigen/CXX11/FixedPoint"
// clang-format on

#include "tensorflow_plugin/src/amd_cpu/util/tstring.h"

namespace amd_cpu_plugin {

// Single precision complex.
typedef std::complex<float> complex64;
// Double precision complex.
typedef std::complex<double> complex128;

// We use Eigen's QInt implementations for our quantized int types.
typedef Eigen::QInt8 qint8;
typedef Eigen::QUInt8 quint8;
typedef Eigen::QInt32 qint32;
typedef Eigen::QInt16 qint16;
typedef Eigen::QUInt16 quint16;

}  // namespace amd_cpu_plugin

static inline Eigen::bfloat16 FloatToBFloat16(float float_val) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  return *reinterpret_cast<Eigen::bfloat16*>(
      reinterpret_cast<uint16_t*>(&float_val));
#else
  return *reinterpret_cast<Eigen::bfloat16*>(
      &(reinterpret_cast<uint16_t*>(&float_val)[1]));
#endif
}

namespace Eigen {
template <>
struct NumTraits<amd_cpu_plugin::tstring>
    : GenericNumTraits<amd_cpu_plugin::tstring> {
  enum {
    RequireInitialization = 1,
    ReadCost = HugeCost,
    AddCost = HugeCost,
    MulCost = HugeCost
  };

  static inline int digits10() { return 0; }

 private:
  static inline amd_cpu_plugin::tstring epsilon();
  static inline amd_cpu_plugin::tstring dummy_precision();
  static inline amd_cpu_plugin::tstring lowest();
  static inline amd_cpu_plugin::tstring highest();
  static inline amd_cpu_plugin::tstring infinity();
  static inline amd_cpu_plugin::tstring quiet_NaN();
};

}  // namespace Eigen

#if defined(_MSC_VER) && !defined(__clang__)
namespace std {
template <>
struct hash<Eigen::half> {
  std::size_t operator()(const Eigen::half& a) const {
    return static_cast<std::size_t>(a.x);
  }
};

template <>
struct hash<Eigen::bfloat16> {
  std::size_t operator()(const Eigen::bfloat16& a) const {
    return hash<float>()(static_cast<float>(a));
  }
};
}  // namespace std

#endif  // _MSC_VER

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_NUMERIC_TYPES_H_
