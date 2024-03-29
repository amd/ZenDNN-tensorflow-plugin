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

#include <vector>
// TensorFlow plug-in headers.
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_kernel_common.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_mempool.h"
#include "tensorflow_plugin/src/amd_cpu/util/errors.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_kernel.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_requires.h"
#include "tensorflow_plugin/src/amd_cpu/util/tensor_format.h"
#include "tensorflow_plugin/src/amd_cpu/util/zen_utils.h"

using zendnn::softmax_forward;

namespace amd_cpu_plugin {

template <typename T>
class ZenSoftmaxOp : public OpKernel {
 public:
  ~ZenSoftmaxOp() {}

  explicit ZenSoftmaxOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, InitZendnnParameters(context, &zendnn_params_));

    string data_format_str;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(context, FormatFromString(data_format_str, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(context, data_format_ == FORMAT_NHWC,
                errors::Unimplemented("ZenDNN Softmax implementation supports "
                                      "NHWC tensor format only for now."));
  }

  void Compute(OpKernelContext* context) override {
    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-DEF: _ZenSoftmax (TF kernel): In Compute!");
    ZenExecutor* ex = ex->getInstance();
    engine eng = ex->getEngine();
    stream s = ex->getStream();

    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    const Tensor& input = context->input(0);
    const int input_dims = input.shape().dims();
    T* input_array = const_cast<T*>(input.template flat<T>().data());
    memory::dims src_dims(input_dims);
    for (int d = 0; d < input_dims; ++d) {
      src_dims[d] = input.shape().dim_size(d);
    }

    // Update the output type.
    ZenTensorType out_type = ZenTensorType::kFloat;

    // Allocating memory for output tensor.
    // Output tensor shape is same as input.
    TensorShape out_shape = input.shape();
    Tensor* output = nullptr;
    zendnnEnv zen_env_obj = readEnv();
    int zen_enable_mempool = zen_env_obj.zenEnableMemPool &&
                             !zendnn_params_.is_eager &&
                             context->expected_output_dtype(0) == DT_FLOAT;
    ZenMemoryPool<T>* zen_pool_buffer = NULL;

    // ZenMemPool Optimization reuse o/p tensors from the pool. By default its
    // enabled, export ZENDNN_ENABLE_MEMPOOL=0 will disable memory pool
    // optimization.
    // Cases where tensors in pool are not free or requested size is more than
    // available tensor size in Pool, control will fall back to default way of
    // allocation i.e. with allocate_output(..).
    // ZenMempool Optimization is not supported by Depthwise Convolution due to
    // performance drop.
    if (zen_enable_mempool) {
      unsigned int thread_id = GetZenTFthreadId(std::this_thread::get_id());
      zen_pool_buffer = ZenMemoryPool<T>::GetZenMemPool(thread_id);
      if (zen_pool_buffer) {
        int status = zen_pool_buffer->AcquireZenPoolTensor(
            context, &output, out_shape, zendnn_params_.out_links,
            zendnn_params_.reset, out_type);
        if (status) {
          zen_enable_mempool = false;
        }
      } else {
        zen_enable_mempool = false;
      }
    }
    if (!zen_enable_mempool) {
      OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
    }

    T* output_array = output->template flat<T>().data();
    memory::dims output_dims = src_dims;
    int axis;

    memory::format_tag layout_type;
    // We use axis to define on which dimension to do softmax. Softmax axis
    // is attached to logical dimensions which always go in a specific
    // order. For softmax it would be N, C, H, W.
    // For a 4D Tensor with softmax axis = 1:
    // {{<physical layout>N,C,H,W}, data_type::f32, format_tag::nhwc};
    // For a 4D Tensor with softmax axis = 3:
    // {{<physical layout>N,H,W,C}, data_type::f32, format_tag::nchw};
    switch (input_dims) {
      case 1:
        layout_type = memory::format_tag::x;  // a
        axis = 0;
        break;
      case 2:
        layout_type = memory::format_tag::nc;  // ab
        axis = 1;
        break;
      case 3:
        layout_type = memory::format_tag::tnc;  // abc
        axis = 2;
        break;
      case 4:
        layout_type = memory::format_tag::nchw;  // abcd
        axis = 3;
        break;
      default:
        OP_REQUIRES_OK(context,
                       errors::Aborted("Input dims must be <= 4 and >=1"));
    }

    // Create softmax memory for src, dst.
    using dt = memory::data_type;
    memory src_memory =
        memory({{src_dims}, dt::f32, layout_type}, eng, input_array);
    memory dst_memory =
        memory({{output_dims}, dt::f32, layout_type}, eng, output_array);

    // Create memory descriptor for src.
    memory::desc src_md = memory::desc({src_dims}, dt::f32, layout_type);

    // Create forward and primitive descriptor for softmax op.
    softmax_forward::desc softmax_fwd_desc =
        softmax_forward::desc(prop_kind::forward_inference, src_md, axis);
    softmax_forward::primitive_desc softmax_fwd_pd =
        softmax_forward::primitive_desc(softmax_fwd_desc, eng);

    auto softmax_fwd = softmax_forward(softmax_fwd_pd);
    net.push_back(softmax_fwd);
    net_args.push_back(
        {{ZENDNN_ARG_SRC, src_memory}, {ZENDNN_ARG_DST, dst_memory}});
    assert(net.size() == net_args.size() && "something is missing");
    for (size_t i = 0; i < net.size(); ++i) {
      net.at(i).execute(s, net_args.at(i));
    }

    // If ZenMemPool Optimization is enabled(default), update the state of
    // memory pool based on input_array address.
    if (zen_env_obj.zenEnableMemPool && !zendnn_params_.is_eager &&
        (input.dtype() == DT_FLOAT) && zen_pool_buffer) {
      zen_pool_buffer->ZenMemPoolFree(context,
                                      reinterpret_cast<float*>(input_array));
    }

    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-DEF: _ZenSoftmax (TF kernel): Compute Is Successful!");
  }

 private:
  TensorFormat data_format_;
  ZendnnParameters zendnn_params_;
};

REGISTER_KERNEL_BUILDER(
    Name("_ZenSoftmax").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    ZenSoftmaxOp<float>);

}  // namespace amd_cpu_plugin
