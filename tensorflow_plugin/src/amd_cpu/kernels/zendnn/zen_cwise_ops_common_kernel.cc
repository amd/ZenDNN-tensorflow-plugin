/*******************************************************************************
 * Modifications Copyright (c) 2024 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 *******************************************************************************/

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0(the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_cwise_ops_common.h"

namespace amd_cpu_plugin {

ZenBinaryOpShared::ZenBinaryOpShared(OpKernelConstruction *ctx, DataType out,
                                     DataType in)
    : OpKernel(ctx) {
  // TODO(plugin): Skip Signature checking, as cannot get input types and output
  // types from OpKernelConstruction in function MatchSignature. It can be done
  // after intergrating graph c api.
  // OP_REQUIRES_OK(ctx, ctx->MatchSignature({in, in}, {out}));
  op_name = ctx->OpName();
  has_attr = ctx->HasAttr("incompatible_shape_error");
  if (has_attr) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("incompatible_shape_error",
                                     &(incompatible_shape_error)));
  }
}

void ZenBinaryOpShared::SetUnimplementedError(OpKernelContext *ctx) {
  ctx->SetStatus(errors::Unimplemented(
      "Broadcast between ", ctx->input(0).shape().DebugString(), " and ",
      ctx->input(1).shape().DebugString(), " is not supported yet."));
}

void ZenBinaryOpShared::SetComputeError(OpKernelContext *ctx) {
  // For speed, errors during compute are caught only via boolean flag, with no
  // associated information.  This is sufficient for now, since the only binary
  // ops that have compute errors are integer division and mod, and the only
  // error they produce is zero division.
  const string &op = op_name;
  if ((op == "Div" || op == "Mod" || op == "FloorMod" || op == "FloorDiv") &&
      DataTypeIsInteger(ctx->input_dtype(0))) {
    ctx->CtxFailure(errors::InvalidArgument("Integer division by zero"));
  } else if ((op == "Pow") && DataTypeIsInteger(ctx->input_dtype(0)) &&
             DataTypeIsSigned(ctx->input_dtype(1))) {
    ctx->CtxFailure(errors::InvalidArgument(
        "Integers to negative integer powers are not allowed"));
  } else {
    ctx->CtxFailure(
        errors::Internal("Unexpected error in binary operator "
                         "(only integer div and mod should have errors)"));
  }
}

ZenBinaryOpShared::ZenBinaryOpState::ZenBinaryOpState(
    OpKernelContext *ctx, const string &op, bool has_attr,
    bool incompatible_shape_error, ZendnnParameters zendnn_params,
    Tensor &cached_buffer_)
    : in0(ctx->input(0)),
      in1(ctx->input(1)),
      bcast(BCast::FromShape(in0.shape()), BCast::FromShape(in1.shape())),
      zendnn_params(zendnn_params),
      in0_reuse(false),
      in1_reuse(false) {
  if (!bcast.IsValid()) {
    if (has_attr && !incompatible_shape_error) {
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));
      result = (op == "NotEqual");
      return;
    }

    ctx->SetStatus(errors::InvalidArgument(
        "Incompatible shapes: ", in0.shape().DebugString(), " vs. ",
        in1.shape().DebugString()));
    return;
  }

  const TensorShape output_shape = BCast::ToShape(bcast.output_shape());
  out_num_elements = output_shape.num_elements();
  in0_num_elements = in0.NumElements();
  in1_num_elements = in1.NumElements();

  // Update the output type.
  ZenTensorType out_type = ZenTensorType::kFloat;

  zendnnEnv zen_env_obj = readEnv();
  int zen_enable_mempool =
      (!zendnn_params.is_eager) ? zen_env_obj.zenEnableMemPool : 0;

  ZenMemoryPool<float> *zen_pool_buffer = NULL;

  // TODO(plugin) : TF-Plugin does not have C APIs for forwarding input to
  // output with shape, therefore, we are defaulting to ZenMemPool Optimization.
  // Output buffer:
  /*  // (1) Reuse any of the input buffers for output
      // (2) If not (1), then get buffer from ZenMemPool
      // (3) If not (1) and (2), then allocate the buffer for output
  if (ctx->forward_input_to_output_with_shape(0, 0, output_shape, &out)) {
    in0_reuse = true;
  } else if (ctx->forward_input_to_output_with_shape(1, 0, output_shape,
                                                     &out)) {
    in1_reuse = true;
  } else */

  // ZenMemPool Optimization reuse o/p tensors from the pool. By default its
  // enabled, export ZENDNN_ENABLE_MEMPOOL=0 will disable memory pool
  // optimization.
  // Cases where tensors in pool are not free or requested size is more than
  // available tensor size in Pool, control will fall back to default way of
  // allocation i.e. with allocate_output(..).
  // ZenMempool Optimization is not supported by Depthwise Convolution due to
  // performance drop.
  if (zen_enable_mempool % MEMPOOL_TYPE) {
    unsigned int thread_id = GetZenTFthreadId(std::this_thread::get_id());
    zen_pool_buffer = ZenMemoryPool<float>::GetZenMemPool(thread_id);
    if (zen_pool_buffer) {
      int status = zen_pool_buffer->AcquireZenPoolTensor(
          ctx, &out, output_shape, zendnn_params.out_links, zendnn_params.reset,
          out_type);
      if (status) {
        zen_enable_mempool = 0;
      }
    } else {
      zen_enable_mempool = 0;
    }
  } else if (zen_enable_mempool) {
    int res = cached_buffer_.NumElements();
    Status state = OkStatus();
    if (res <= 0 || res != output_shape.num_elements()) {
      state =
          ctx->allocate_temp(DataType::DT_FLOAT, output_shape, &cached_buffer_);
    }
    if (state != OkStatus()) {
      zen_enable_mempool = 0;
    } else {
      out = &cached_buffer_;
      ctx->set_output(0, *out);
    }
  }
  if (!zen_enable_mempool) {
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &out));
  }
  ndims = static_cast<int>(bcast.x_reshape().size());
}

#define REGISTER_CWISE_KERNELS(OP, D, N, F, T)                               \
  REGISTER_KERNEL_BUILDER(Name(N).Device(DEVICE_##D).TypeConstraint<T>("T"), \
                          OP<F<T>>);

REGISTER_CWISE_KERNELS(ZenBinaryOp, CPU, "_ZenAdd", functor::add, float);
REGISTER_CWISE_KERNELS(ZenBinaryOp, CPU, "_ZenAddV2", functor::add, float);
REGISTER_CWISE_KERNELS(ZenBinaryOp, CPU, "_ZenSub", functor::sub, float);
REGISTER_CWISE_KERNELS(ZenBinaryOp, CPU, "_ZenMul", functor::mul, float);
REGISTER_CWISE_KERNELS(ZenBinaryOp, CPU, "_ZenMaximum", functor::maximum,
                       float);
REGISTER_CWISE_KERNELS(ZenBinaryOp, CPU, "_ZenSquaredDifference",
                       functor::squared_difference, float);
#undef REGISTER_CWISE_KERNELS
}  // end namespace amd_cpu_plugin
