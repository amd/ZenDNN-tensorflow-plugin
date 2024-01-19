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

#include <memory>
#include <vector>
// TensorFlow plug-in headers.
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_kernel_common.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_mempool.h"
#include "tensorflow_plugin/src/amd_cpu/util/errors.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_kernel.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_requires.h"
#include "tensorflow_plugin/src/amd_cpu/util/overflow.h"
#include "tensorflow_plugin/src/amd_cpu/util/plugin_tensor.h"
#include "tensorflow_plugin/src/amd_cpu/util/register_types.h"
#include "tensorflow_plugin/src/amd_cpu/util/zen_utils.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace amd_cpu_plugin {

template <typename T>
class ZenReshapeOp : public OpKernel {
 public:
  explicit ZenReshapeOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, InitZendnnParameters(context, &zendnn_params_));
  }

  void Compute(OpKernelContext* context) override {
    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-DEF: _ZenReshape (TF kernel): In Compute!");

    const Tensor& input = context->input(0);
    const Tensor& sizes = context->input(1);
    // Preliminary validation of sizes.
    OP_REQUIRES(
        context,
        (TensorShapeUtils::IsVector(sizes.shape()) ||
         // TODO(plugin): Disallow legacy use of scalars to represent shape.
         TensorShapeUtils::IsScalar(sizes.shape())),
        errors::InvalidArgument("sizes input must be 1-D, not ",
                                sizes.shape().DebugString()));

    // Compute the output shape.  Determine product of specified
    // dimensions, and find the index of the unspecified one.
    TensorShape shape;
    int64_t product = 1;
    int unknown_index = -1;
    bool sizes_has_zero_dim;
    switch (sizes.dtype()) {
      case DT_INT32:
        OP_REQUIRES_OK(context,
                       ValidateSizes<int32>(sizes, &product, &unknown_index,
                                            &shape, &sizes_has_zero_dim));
        break;
      case DT_INT64:
        OP_REQUIRES_OK(context,
                       ValidateSizes<int64_t>(sizes, &product, &unknown_index,
                                              &shape, &sizes_has_zero_dim));
        break;
      default:
        context->CtxFailure(errors::InvalidArgument(
            "desired shape must be a DT_INT32 or DT_INT64 vector, not a ",
            DataTypeString(sizes.dtype())));
        return;
    }
    if (unknown_index != -1) {
      int64_t input_num_elements = 1;
      bool input_has_zero_dim = false;
      for (int dim = 0; dim < input.dims(); dim++) {
        // For zero dimension, we don't count it into `input_num_elements`
        // unless `sizes` has no zero dimension, so we are still able to
        // infer shapes for other dimensions.
        if (input.dim_size(dim) > 0 || !sizes_has_zero_dim) {
          input_num_elements *= input.dim_size(dim);
        } else {
          input_has_zero_dim = true;
        }
      }

      const int64_t missing = input_num_elements / product;
      if (!input_has_zero_dim) {
        OP_REQUIRES(
            context, product * missing == input_num_elements,
            errors::InvalidArgument(
                "Input to reshape is a tensor with ", input_num_elements,
                " values, but the requested shape requires a multiple of ",
                product));
      }
      shape.set_dim(unknown_index, missing);
    }
    OP_REQUIRES(context, shape.num_elements() == input.NumElements(),
                errors::InvalidArgument("Input to reshape is a tensor with ",
                                        input.NumElements(),
                                        " values, but the requested shape has ",
                                        shape.num_elements()));

    // Actually produce the reshaped output.
    Tensor output(input.dtype());
    CHECK(output.CopyFrom(input, shape));
    context->set_output(0, output);

    // If ZenMemPool is enabled then, update the buffer use status if input
    // buffer is used for the output buffer.
    zendnnEnv zen_env_obj = readEnv();
    ZenMemoryPool<T>* zen_pool_buffer = NULL;
    if ((zen_env_obj.zenEnableMemPool % MEMPOOL_TYPE) &&
        !zendnn_params_.is_eager) {
      unsigned int thread_id = GetZenTFthreadId(std::this_thread::get_id());
      zen_pool_buffer = ZenMemoryPool<T>::GetZenMemPool(thread_id);
      if (zen_pool_buffer) {
        auto in0_ptr = const_cast<T*>(input.template flat<T>().data());
        zen_pool_buffer->ZenMemPoolUpdateTensorPtrStatus(
            context, in0_ptr, zendnn_params_.out_links, zendnn_params_.reset);
      }
    }

    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-DEF: _ZenReshape (TF kernel): Compute Is Successful!");
  }

 private:
  ZendnnParameters zendnn_params_;
  template <typename Tshape>
  Status ValidateSizes(const Tensor& sizes, int64_t* product,
                       int* unknown_index, TensorShape* shape,
                       bool* has_zero_dim) {
    *product = 1;
    *unknown_index = -1;
    *has_zero_dim = false;
    const int64_t num_dims = sizes.NumElements();
    auto Svec = sizes.flat<Tshape>();
    for (int d = 0; d < num_dims; ++d) {
      const Tshape size = Svec(d);
      if (size == -1) {
        if (*unknown_index != -1) {
          return errors::InvalidArgument(
              "Only one input size may be -1, not both ", *unknown_index,
              " and ", d);
        }
        *unknown_index = d;
        shape->AddDim(1);
      } else if (size < 0) {
        return errors::InvalidArgument("Size ", d,
                                       " must be non-negative, not ", size);
      } else if (size == 0) {
        // We don't include zero-sized dimension in product, so that we can
        // still calculate number of elements for non-zero-sized dimensions and
        // therefore infer their shapes.
        shape->AddDim(size);
        *has_zero_dim = true;
      } else {
        if (MultiplyWithoutOverflow(shape->num_elements(), size) < 0) {
          string msg;
          for (int ii = 0; ii < num_dims; ++ii) {
            if (ii != 0) {
              strings::StrAppend(&msg, ", ");
            }
            strings::StrAppend(&msg, Svec(ii));
          }
          return errors::InvalidArgument("Shape [", msg,
                                         "] has too many elements");
        }
        shape->AddDim(size);
        (*product) *= size;
      }
    }
    return OkStatus();
  }
};

#define REGISTER_RESHAPE_KERNEL(TYPE)                    \
  REGISTER_KERNEL_BUILDER(Name("_ZenReshape")            \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<TYPE>("T") \
                              .HostMemory("shape"),      \
                          ZenReshapeOp<TYPE>);
TF_CALL_NUMBER_TYPES(REGISTER_RESHAPE_KERNEL)
#undef REGISTER_RESHAPE_KERNEL

}  // namespace amd_cpu_plugin
