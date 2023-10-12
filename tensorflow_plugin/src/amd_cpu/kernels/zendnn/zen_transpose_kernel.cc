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

// Standard headers.
#include <limits>
// TensorFlow plug-in headers.
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_kernel_common.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_mempool.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_transpose_functor.h"
#include "tensorflow_plugin/src/amd_cpu/util/bounds_check.h"
#include "tensorflow_plugin/src/amd_cpu/util/errors.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_requires.h"
#include "tensorflow_plugin/src/amd_cpu/util/register_types.h"
#include "tensorflow_plugin/src/amd_cpu/util/tensor_format.h"
#include "tensorflow_plugin/src/amd_cpu/util/zen_utils.h"

namespace amd_cpu_plugin {

template <typename T, bool is_conjugate = false>
class ZenTransposeOp : public OpKernel {
 public:
  explicit ZenTransposeOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, InitZendnnParameters(context, &zendnn_params_));
  }

  // output = ZenTransposeOp(T<any> input, T<int32> perm) takes a tensor
  // of type T and rank N, and a permutation of 0, 1, ..., N-1. It
  // shuffles the dimensions of the input tensor according to permutation.
  //
  // Specifically, the returned tensor output meets the following condition:
  // 1) output.dims() == input.dims();
  // 2) output.dim_size(i) == input.dim_size(perm[i]);
  // 3) output.tensor<T, N>(i_0, i_1, ..., i_N-1) ==
  //      input.tensor<T, N>(j_0, j_1, ..., j_N-1),
  //    where i_s == j_{perm[s]}
  //
  // REQUIRES: perm is a vector of int32.
  // REQUIRES: input.dims() == perm.size().
  // REQUIRES: perm is a permutation.
  void Compute(OpKernelContext *context) override {
    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-DEF: _ZenTranspose (TF kernel): In Compute!");

    const Tensor &input = context->input(0);
    const Tensor &perm = context->input(1);

    // Preliminary validation of sizes.
    OP_REQUIRES(context, TensorShapeUtils::IsVector(perm.shape()),
                errors::InvalidArgument("perm must be a vector, not ",
                                        perm.shape().DebugString()));

    // Although Tperm may be an int64 type, an int32 is sufficient to hold
    // dimension range values, so the narrowing here should be safe.
    std::vector<int32> permutation;
    const int dims = input.dims();
    if (perm.dtype() == DT_INT32) {
      OP_REQUIRES_OK(context, internal::PermutationHelper<int32>(perm, dims,
                                                                 &permutation));
    } else {
      OP_REQUIRES_OK(context, internal::PermutationHelper<int64>(perm, dims,
                                                                 &permutation));
    }
    TensorShape shape;

    // Check whether permutation is a permutation of integers of [0 .. dims).
    gtl::InlinedVector<bool, 8> bits(dims);
    bool is_identity = true;
    for (int idx = 0; idx < dims; ++idx) {
      const int32 value = permutation[idx];
      OP_REQUIRES(
          context, 0 <= value && value < dims,
          errors::InvalidArgument(value, " is out of range [0 .. ", dims, ")"));
      bits[value] = true;
      const auto dim_size = input.dim_size(value);
      shape.AddDim(dim_size);
      if (value != idx) {
        is_identity = false;
      }
    }
    for (int i = 0; i < dims; ++i) {
      OP_REQUIRES(
          context, bits[i],
          errors::InvalidArgument(i, " is missing from {",
                                  absl::StrJoin(permutation, ","), "}."));
    }

    // 0-D, 1-D, and identity transposes do nothing.
    if (!is_conjugate && (dims <= 1 || is_identity)) {
      context->set_output(0, input);
      return;
    } else if (!is_conjugate && internal::NonSingletonDimensionsAlign(
                                    input.shape(), permutation)) {
      Tensor output;
      OP_REQUIRES(context, output.CopyFrom(input, shape),
                  errors::Unknown("Error reshaping Tensor."));
      context->set_output(0, output);
      return;
    }

    // Update the output type.
    ZenTensorType out_type = ZenTensorType::kFloat;

    // Output tensor.
    Tensor *output = nullptr;
    zendnnEnv zen_env_obj = readEnv();
    int zen_enable_mempool = zen_env_obj.zenEnableMemPool &&
                             !zendnn_params_.is_eager &&
                             context->expected_output_dtype(0) == DT_FLOAT;
    ZenMemoryPool<T> *zen_pool_buffer = NULL;

    // ZenMemPool Optimization reuse o/p tensors from the pool. By default its
    // enabled, export ZENDNN_ENABLE_MEMPOOL=0 will disable memory pool
    // optimization.
    // Cases where tensors in pool are not free or requested size is more than
    // available tensor size in Pool, control will fall back to default way of
    // allocation i.e. with allocate_output(..).
    if (zen_enable_mempool) {
      unsigned int thread_id = GetZenTFthreadId(std::this_thread::get_id());
      zen_pool_buffer = ZenMemoryPool<T>::GetZenMemPool(thread_id);
      if (zen_pool_buffer) {
        int status = zen_pool_buffer->AcquireZenPoolTensor(
            context, &output, shape, zendnn_params_.out_links,
            zendnn_params_.reset, out_type);
        if (status) {
          zen_enable_mempool = false;
        }
      } else {
        zen_enable_mempool = false;
      }
    }
    if (!zen_enable_mempool) {
      OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output));
    }

    if (shape.num_elements() > 0) {
      OP_REQUIRES_OK(context, DoTranspose<T, is_conjugate>(
                                  context, input, permutation, output));
    }

    // If ZenMemPool Optimization is enabled(default), update the state of
    // memory pool based on input_array address.
    if (zen_env_obj.zenEnableMemPool && !zendnn_params_.is_eager &&
        (input.dtype() == DT_FLOAT) && zen_pool_buffer) {
      float *input_array =
          const_cast<float *>(input.template flat<float>().data());
      zen_pool_buffer->ZenMemPoolFree(context,
                                      const_cast<float *>(input_array));
    }

    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-DEF: _ZenTranspose (TF kernel): Compute Is Successful!");
  }

 private:
  /* ZenDNN specific */
  ZendnnParameters zendnn_params_;
};

// inv = ZenInvertPermutationOp(T<int32/int64> p) takes a permutation of
// integers 0, 1, ..., n - 1 and returns the inverted
// permutation of p. I.e., inv[p[i]] == i, for i in [0 .. n).
//
// REQUIRES: input is a vector of int32 or int64.
// REQUIRES: input is a permutation of 0, 1, ..., n-1.
template <typename T>
class ZenInvertPermutationOp : public OpKernel {
 public:
  explicit ZenInvertPermutationOp(OpKernelConstruction *context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, InitZendnnParameters(context, &zendnn_params_));
  }

  void Compute(OpKernelContext *context) override {
    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-DEF: _ZenInvertPermutation (TF kernel): In Compute!");

    const Tensor &input = context->input(0);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(input.shape()),
        errors::InvalidArgument("invert_permutation expects a 1D vector."));
    auto Tin = input.vec<T>();
    OP_REQUIRES(context,
                FastBoundsCheck(Tin.size(), std::numeric_limits<int32>::max()),
                errors::InvalidArgument("permutation of nonnegative int32s "
                                        "must have <= int32 max elements"));
    const T N = static_cast<T>(Tin.size());  // Safe: bounds-checked above.

    // Output tensor.
    Tensor *output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    auto Tout = output->vec<T>();
    std::fill_n(Tout.data(), N, -1);
    for (int i = 0; i < N; ++i) {
      const T d = internal::SubtleMustCopy(Tin(i));
      OP_REQUIRES(context, FastBoundsCheck(d, N),
                  errors::InvalidArgument(d, " is not between 0 and ", N));
      OP_REQUIRES(context, Tout(d) == -1,
                  errors::InvalidArgument(d, " is duplicated in the input."));
      Tout(d) = i;
    }

    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-DEF: _ZenInvertPermutation (TF kernel): Compute Is "
               "Successful!");
  }

 private:
  /* ZenDNN specific */
  ZendnnParameters zendnn_params_;
};

REGISTER_KERNEL_BUILDER(
    Name("_ZenInvertPermutation").Device(DEVICE_CPU).TypeConstraint<int32>("T"),
    ZenInvertPermutationOp<int32>);
REGISTER_KERNEL_BUILDER(
    Name("_ZenInvertPermutation").Device(DEVICE_CPU).TypeConstraint<int64>("T"),
    ZenInvertPermutationOp<int64>);

#define REGISTER_TRANSPOSE_KERNELS(TYPE)                                  \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("_ZenTranspose").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      ZenTransposeOp<TYPE, false>);                                       \
  REGISTER_KERNEL_BUILDER(Name("_ZenConjugateTranspose")                  \
                              .Device(DEVICE_CPU)                         \
                              .TypeConstraint<TYPE>("T"),                 \
                          ZenTransposeOp<TYPE, true>);
TF_CALL_ALL_TYPES(REGISTER_TRANSPOSE_KERNELS)
#undef REGISTER_TRANSPOSE_KERNELS

}  // namespace amd_cpu_plugin
