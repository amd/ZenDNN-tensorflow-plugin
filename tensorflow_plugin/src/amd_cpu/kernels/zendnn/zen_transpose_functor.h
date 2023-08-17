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

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_TRANSPOSE_FUNCTOR_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_TRANSPOSE_FUNCTOR_H_

// Standard headers.
#include <vector>

// TensorFlow plug-in headers.
#include "tensorflow_plugin/src/amd_cpu/util/gtl/array_slice.h"
#include "tensorflow_plugin/src/amd_cpu/util/gtl/inlined_vector.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_kernel.h"
#include "tensorflow_plugin/src/amd_cpu/util/plugin_tensor.h"
#include "tensorflow_plugin/src/amd_cpu/util/tensor_shape.h"
#include "tensorflow_plugin/src/amd_cpu/util/tensor_types.h"

namespace amd_cpu_plugin {

typedef Eigen::ThreadPoolDevice CPUDevice;

// Implementation details.
namespace internal {

// If all non-singleton dimensions remain in ascending order, the shuffled
// singletons can be transposed by a reshape, saving a memory allocation & copy.
// |permutation| must be a permutation of {0, .., input_shape.dims() - 1}.
// That is, for all i, 0 <= perm[i] < input_shape.dims().
// In practice, this is checked in TransposeOp::Compute prior to calling this
// function, and the function sits here to facilitate unit testing.
inline bool NonSingletonDimensionsAlign(const TensorShape& input_shape,
                                        const std::vector<int32>& permutation) {
  int last_nonsingleton_perm_dim = -1;
  for (int perm_dim : permutation) {
    if (input_shape.dim_size(perm_dim) == 1) {
      continue;
    }
    if (perm_dim < last_nonsingleton_perm_dim) {
      return false;
    }
    last_nonsingleton_perm_dim = perm_dim;
  }
  return true;
}

// Uses Eigen to transpose.
template <typename Device, typename T, int NDIMS>
void TransposeUsingEigen(const Device& d, const Tensor& in,
                         const gtl::ArraySlice<int32> perm, bool conjugate,
                         Tensor* out) {
  Eigen::array<int, NDIMS> p;
  for (int i = 0; i < NDIMS; ++i) {
    p[i] = perm[i];
  }
  auto x = typename TTypes<T, NDIMS>::ConstTensor(
      reinterpret_cast<const T*>(in.tensor_data().data()),
      in.shape().AsEigenDSizes<NDIMS>());
  auto y = typename TTypes<T, NDIMS>::Tensor(
      reinterpret_cast<T*>(const_cast<char*>(out->tensor_data().data())),
      out->shape().AsEigenDSizes<NDIMS>());
  if (conjugate) {
    y.device(d) = x.conjugate().shuffle(p);
  } else {
    y.device(d) = x.shuffle(p);
  }
}

template <typename Tperm>
Status PermutationHelper(const Tensor& perm, const int dims,
                         std::vector<int32>* permutation) {
  auto Vperm = perm.vec<Tperm>();
  if (dims != Vperm.size()) {
    return errors::InvalidArgument("transpose expects a vector of size ", dims,
                                   ". But input(1) is a vector of size ",
                                   Vperm.size());
  }
  // Using volatile instead of SubtleMustCopy here so that the asynchrony
  // boundary is permutation.
  const volatile Tperm* perm_begin =
      reinterpret_cast<const volatile Tperm*>(Vperm.data());
  *permutation = std::vector<int32>(perm_begin, perm_begin + dims);

  return OkStatus();
}

}  // namespace internal

// Transpose tensor 'in' into tensor 'out' according to dimension
// permutation 'perm'.
//
// REQUIRES: in.dtype() == out->dtype()
// REQUIRES: in.dims() == out->dims()
// REQUIRES: in.dims() == perm.size()
// REQUIRES: in.dim_size(perm[i]) == out->dim_size(i)
template <typename T, bool conjugate>
Status DoTranspose(OpKernelContext* context, const Tensor& in,
                   gtl::ArraySlice<int32> perm, Tensor* out) {
  auto in_dims = in.dims();
  if (in_dims < 2) {
    return OkStatus();
  }
  if (false) {
    // TODO(plugin): ZenDNN's implementation.
  } else {
    const CPUDevice& d = context->eigen_cpu_device();
    switch (in_dims) {
      case 2:
        internal::TransposeUsingEigen<CPUDevice, T, 2>(d, in, perm, conjugate,
                                                       out);
        break;
      case 3:
        internal::TransposeUsingEigen<CPUDevice, T, 3>(d, in, perm, conjugate,
                                                       out);
        break;
      case 4:
        internal::TransposeUsingEigen<CPUDevice, T, 4>(d, in, perm, conjugate,
                                                       out);
        break;
      case 5:
        internal::TransposeUsingEigen<CPUDevice, T, 5>(d, in, perm, conjugate,
                                                       out);
        break;
      case 6:
        internal::TransposeUsingEigen<CPUDevice, T, 6>(d, in, perm, conjugate,
                                                       out);
        break;
      case 7:
        internal::TransposeUsingEigen<CPUDevice, T, 7>(d, in, perm, conjugate,
                                                       out);
        break;
      case 8:
        internal::TransposeUsingEigen<CPUDevice, T, 8>(d, in, perm, conjugate,
                                                       out);
        break;
      default:
        CHECK(false) << "Max supported dim number is 8, got " << in_dims;
        break;
    }
  }
  return OkStatus();
}

}  // namespace amd_cpu_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_TRANSPOSE_FUNCTOR_H_
