/*******************************************************************************
 * Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
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

// TensorFlow plug-in headers.
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_kernel_common.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_zendnnl_utils.h"
#include "tensorflow_plugin/src/amd_cpu/util/errors.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_kernel.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_requires.h"
#include "tensorflow_plugin/src/amd_cpu/util/register_types.h"
#include "tensorflow_plugin/src/amd_cpu/util/tensor_format.h"
#include "tensorflow_plugin/src/amd_cpu/util/zen_utils.h"

// ZenDNNL Low Overhead API headers
#include "lowoha_operators/softmax/lowoha_softmax.hpp"
#include "lowoha_operators/softmax/lowoha_softmax_common.hpp"
#include "lowoha_operators/softmax/lowoha_softmax_utils.hpp"

namespace amd_cpu_plugin {

// ZenDNNL Softmax implementation using Low Overhead API (softmax_direct).
template <typename T>
bool TryExecuteZenDNNLSoftmax(const Tensor& input, Tensor* output, int axis) {
  try {
    using namespace zendnnl::lowoha::softmax;
    using namespace zendnnl::common;

    // Get pointers to TensorFlow tensor data.
    T* input_data = const_cast<T*>(input.flat<T>().data());
    T* output_data = output->flat<T>().data();

    const int input_dims = input.shape().dims();

    // Validate number of dimensions
    if (input_dims <= 0 || input_dims > SOFTMAX_MAX_NDIMS) {
      return false;
    }

    // Setup softmax parameters
    softmax_params params;
    params.log_softmax = false;  // Regular softmax (not log-softmax)

    // Set data types based on input type
    if (std::is_same<T, float>::value) {
      params.src_dt = data_type_t::f32;
      params.dst_dt = data_type_t::f32;
    } else if (std::is_same<T, Eigen::bfloat16>::value) {
      params.src_dt = data_type_t::bf16;
      params.dst_dt = data_type_t::bf16;
    } else {
      return false;
    }

    // Convert TensorFlow shape to uint64_t array
    uint64_t shape[SOFTMAX_MAX_NDIMS];
    auto input_dim_sizes = input.shape().dim_sizes();
    for (int d = 0; d < input_dims; ++d) {
      shape[d] = static_cast<uint64_t>(input_dim_sizes[d]);
    }

    // Use setup_softmax_shape from ZenDNN library to populate all params
    // This calculates batch, axis_dim, inner_size, and stores original shape
    status_t status = setup_softmax_shape(params, shape, input_dims, axis);
    if (status != status_t::success) {
      return false;
    }

    // Execute softmax using ZenDNNL Low Overhead API
    status = softmax_direct(input_data, output_data, params);

    return (status == status_t::success);

  } catch (const std::exception& e) {
    return false;
  }
}

// Specialized versions for float and bfloat16
template bool TryExecuteZenDNNLSoftmax<float>(const Tensor&, Tensor*, int);
template bool TryExecuteZenDNNLSoftmax<Eigen::bfloat16>(const Tensor&, Tensor*,
                                                        int);

template <typename T>
class ZenSoftmaxOp : public OpKernel {
 public:
  ~ZenSoftmaxOp() {}

  explicit ZenSoftmaxOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format_str = "";
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(context, FormatFromString(data_format_str, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(context, data_format_ == FORMAT_NHWC,
                errors::Unimplemented("ZenDNN Softmax implementation supports "
                                      "NHWC tensor format only for now."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const int input_dims = input.shape().dims();

    // Allocating memory for output tensor.
    // Output tensor shape is same as input.
    TensorShape out_shape = input.shape();
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    // Softmax axis is attached to logical dimensions (last dimension)
    int axis = -1;

    // Execute ZenDNNL softmax using Low Overhead API
    bool zendnnl_success = TryExecuteZenDNNLSoftmax<T>(input, output, axis);

    OP_REQUIRES(context, zendnnl_success,
                errors::Internal("ZenDNNL Softmax execution failed"));
  }

 private:
  TensorFormat data_format_ = TensorFormat::FORMAT_NHWC;
};

#define REGISTER_SOFTMAX_KERNELS(TYPE)                                  \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("_ZenSoftmax").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      ZenSoftmaxOp<TYPE>);

TF_CALL_float(REGISTER_SOFTMAX_KERNELS);
TF_CALL_bfloat16(REGISTER_SOFTMAX_KERNELS);
#undef REGISTER_SOFTMAX_KERNELS

}  // namespace amd_cpu_plugin
