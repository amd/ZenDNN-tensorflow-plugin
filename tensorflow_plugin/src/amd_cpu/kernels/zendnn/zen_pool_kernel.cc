/*******************************************************************************
 * Modifications Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 ******************************************************************************/

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// TensorFlow plug-in headers.
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/pooling_ops_common.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_kernel_common.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_mempool.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_requires.h"
#include "tensorflow_plugin/src/amd_cpu/util/register_types.h"
#include "tensorflow_plugin/src/amd_cpu/util/tensor_format.h"
#include "tensorflow_plugin/src/amd_cpu/util/zen_utils.h"

#define NEW_API 1

using zendnn::pooling_forward;
using zendnn::reorder;

namespace amd_cpu_plugin {

template <typename T, bool is_maxpool>
class ZenPoolOp : public OpKernel {
 public:
  explicit ZenPoolOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(
        context, ksize_.size() == 4,
        errors::InvalidArgument("Kernel size field must specify 4 dimensions"));

    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument(
                    "Sliding window stride field must specify 4 dimensions"));

    string padding_str;
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_str));
    if (padding_str == "VALID") {
      padding_ = Padding::VALID;
    } else if (padding_str == "SAME") {
      padding_ = Padding::SAME;
    } else {
      padding_ = Padding::EXPLICIT;
    }
    if (context->HasAttr("explicit_paddings")) {
      OP_REQUIRES_OK(
          context, context->GetAttr("explicit_paddings", &explicit_paddings_));
    }

    string data_format_str;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(context, FormatFromString(data_format_str, &data_format_),
                errors::InvalidArgument("Invalid data format"));

    OP_REQUIRES_OK(context, InitZendnnParameters(context, &zendnn_params_));
  }

  void Compute(OpKernelContext *context) override {
    zendnnInfo(ZENDNN_FWKLOG, "ZEN-OP-DEF: _ZenPool (TF kernel): In Compute!");

    int data_format = (data_format_ == FORMAT_NCHW) ? 1 : 0;

    const Tensor &input = context->input(0);
    const T *input_array = const_cast<T *>(input.template flat<T>().data());

    PoolParameters params{context,
                          ksize_,
                          stride_,
                          padding_,
                          /*explict padding*/ {},
                          data_format_,
                          input.shape()};
    TensorShape out_shape = params.forward_output_shape();
    // Update the output type.
    ZenTensorType out_type = ZenTensorType::kFloat;

    // Output tensor.
    Tensor *output = nullptr;
    zendnnEnv zen_env_obj = readEnv();
    int zen_enable_mempool =
        zendnn_params_.is_eager ? 0 : zen_env_obj.zenEnableMemPool;
    ZenMemoryPool<T> *zen_pool_buffer = NULL;

    // ZenMemPool optimization reuse o/p tensors from the pool. By default its
    // enabled, export ZENDNN_ENABLE_MEMPOOL=0 will disable memory pool
    // optimization.
    // Cases where tensors in pool are not free or requested size is more than
    // available tensor size in Pool, control will fall back to default way of
    // allocation i.e. with allocate_output(..).
    if (zen_enable_mempool % MEMPOOL_TYPE) {
      unsigned int thread_id = GetZenTFthreadId(std::this_thread::get_id());
      zen_pool_buffer = ZenMemoryPool<T>::GetZenMemPool(thread_id);
      if (zen_pool_buffer) {
        int status = zen_pool_buffer->AcquireZenPoolTensor(
            context, &output, out_shape, zendnn_params_.out_links,
            zendnn_params_.reset, out_type);
        if (status) {
          zen_enable_mempool = 0;
        }
      } else {
        zen_enable_mempool = 0;
      }
    }
    if (!(zen_enable_mempool % MEMPOOL_TYPE)) {
      OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
    }

    T *output_array = const_cast<T *>(output->template flat<T>().data());

    const int image_height = params.tensor_in_rows;
    const int image_width = params.tensor_in_cols;

    int stride_h, stride_w, filter_height, filter_width;
    int padding_h_top, padding_h_bottom, padding_w_left, padding_w_right;

    stride_h = stride_[1];
    stride_w = stride_[2];

    filter_height = ksize_[1];
    filter_width = ksize_[2];

    // TODO(plugin): Define new function compute_padding for this as its used at
    // multiple places.
    if (!(padding_ == SAME)) {
      padding_h_top = padding_h_bottom = padding_w_left = padding_w_right = 0;
    } else {
      int total_pad_h, total_pad_w;
      int mod_h, mod_w;
      mod_h = image_height % stride_h;
      mod_w = image_width % stride_w;

      total_pad_h =
          std::max(filter_height - (mod_h == 0 ? stride_h : mod_h), 0);
      padding_h_top =
          (total_pad_h / 2);  // Integer division equivalent to floor.
      padding_h_bottom = total_pad_h - padding_h_top;

      total_pad_w = std::max(filter_width - (mod_w == 0 ? stride_w : mod_w), 0);
      padding_w_left =
          (total_pad_w / 2);  // Integer division equivalent to floor.
      padding_w_right = total_pad_w - padding_w_left;
    }

#if NEW_API
    bool blocked = zen_env_obj.zenConvAlgo == zenConvAlgoType::DIRECT1 &&
                   !zendnn_params_.is_eager;

    if (blocked) {
      zendnnInfo(ZENDNN_FWKLOG,
                 "_ZenPool (TF kernel): New API for BLOCKED POOLING _ZenPool");
      using tag = memory::format_tag;
      using dt = memory::data_type;
      ZenExecutor *ex = ex->getInstance();
      engine eng = ex->getEngine();
      stream s = ex->getStream();
      std::vector<primitive> net;
      std::vector<std::unordered_map<int, memory>> net_args;

      memory::dim out_height, out_width;
      out_height = (params.tensor_in_rows - params.window_rows + padding_h_top +
                    padding_h_bottom) /
                       params.row_stride +
                   1;
      out_width = (params.tensor_in_cols - params.window_cols + padding_w_left +
                   padding_w_right) /
                      params.col_stride +
                  1;

      memory::dims pool_src_tz = {params.tensor_in_batch, params.depth,
                                  params.tensor_in_rows, params.tensor_in_cols};
      memory::dims pool_dst_tz = {params.tensor_in_batch, params.depth,
                                  out_height, out_width};
      memory::dims pool_kernel = {params.window_rows, params.window_cols};
      memory::dims pool_strides = {params.row_stride, params.col_stride};
      memory::dims pool_padding_l = {padding_h_top, padding_w_left};
      memory::dims pool_padding_r = {padding_h_bottom, padding_w_right};

      memory pool_src_memory;
      memory pool_dst_memory, pool_dst_memory_new;

      if (zendnn_params_.reorder_before) {
        pool_src_memory = memory({{pool_src_tz}, dt::f32, tag::nhwc}, eng,
                                 const_cast<T *>(input_array));
      } else {
        pool_src_memory = memory({{pool_src_tz}, dt::f32, tag::aBcd8b}, eng,
                                 const_cast<T *>(input_array));
      }
      if (zendnn_params_.reorder_after) {
        pool_dst_memory = memory({{pool_dst_tz}, dt::f32, tag::aBcd8b}, eng);
        pool_dst_memory_new = memory({{pool_dst_tz}, dt::f32, tag::nhwc}, eng,
                                     reinterpret_cast<T *>(output_array));
      } else {
        pool_dst_memory = memory({{pool_dst_tz}, dt::f32, tag::aBcd8b}, eng,
                                 reinterpret_cast<T *>(output_array));
        pool_dst_memory_new = memory({{pool_dst_tz}, dt::f32, tag::aBcd8b}, eng,
                                     reinterpret_cast<T *>(output_array));
      }

      memory::desc pool_src_md =
          memory::desc({pool_src_tz}, dt::f32, tag::aBcd8b);
      memory::desc pool_dst_md =
          memory::desc({pool_dst_tz}, dt::f32, tag::aBcd8b);

      algorithm pooling_algo =
          is_maxpool ? algorithm::pooling_max : algorithm::pooling_avg;
      // [Create pooling primitive]
      pooling_forward::desc pool_desc = pooling_forward::desc(
          prop_kind::forward_inference, pooling_algo, pool_src_md, pool_dst_md,
          pool_strides, pool_kernel, pool_padding_l, pool_padding_r);

      pooling_forward::primitive_desc pool_pd =
          pooling_forward::primitive_desc(pool_desc, eng);

      memory pool1_src_memory = pool_src_memory;
      if (pool_pd.src_desc() != pool_src_memory.get_desc()) {
        pool1_src_memory = memory(pool_pd.src_desc(), eng);
        if (zendnn_params_.reorder_before) {
          net.push_back(reorder(pool_src_memory, pool1_src_memory));
          net_args.push_back({{ZENDNN_ARG_SRC, pool_src_memory},
                              { ZENDNN_ARG_DST,
                                pool1_src_memory }});
        }
      }

      net.push_back(pooling_forward(pool_pd));
      net_args.push_back({{ZENDNN_ARG_SRC, pool1_src_memory},
                          { ZENDNN_ARG_DST,
                            pool_dst_memory }});

      // [Execute model]
      assert(net.size() == net_args.size() && "something is missing");
      for (size_t i = 0; i < net.size(); ++i) {
        net.at(i).execute(s, net_args.at(i));
      }
      if (zendnn_params_.reorder_after) {
        reorder(pool_dst_memory, pool_dst_memory_new)
            .execute(s, pool_dst_memory, pool_dst_memory_new);
      }
    } else {
      // TODO(ZenDNN): Create ZenDNN API for ZenDNN Library pooling.
      zendnnInfo(ZENDNN_FWKLOG,
                 "_ZenPool (TF kernel): New API for NHWC POOLING _ZenPool");
      if (is_maxpool) {
        max_pooling(input_array, params.tensor_in_batch, params.depth,
                    params.tensor_in_rows, params.tensor_in_cols,
                    params.window_rows, params.window_cols, params.row_stride,
                    params.col_stride, padding_h_top, padding_h_bottom,
                    padding_w_left, padding_w_right, output_array, data_format);
      } else {
        avg_pooling(input_array, params.tensor_in_batch, params.depth,
                    params.tensor_in_rows, params.tensor_in_cols,
                    params.window_rows, params.window_cols, params.row_stride,
                    params.col_stride, padding_h_top, padding_h_bottom,
                    padding_w_left, padding_w_right, output_array, data_format);
      }
    }
#else
    if (is_maxpool) {
      max_pooling(input_array, params.tensor_in_batch, params.depth,
                  params.tensor_in_rows, params.tensor_in_cols,
                  params.window_rows, params.window_cols, params.row_stride,
                  params.col_stride, padding_h_top, padding_h_bottom,
                  padding_w_left, padding_w_right, output_array, data_format);
    } else {
      avg_pooling(input_array, params.tensor_in_batch, params.depth,
                  params.tensor_in_rows, params.tensor_in_cols,
                  params.window_rows, params.window_cols, params.row_stride,
                  params.col_stride, padding_h_top, padding_h_bottom,
                  padding_w_left, padding_w_right, output_array, data_format);
    }
#endif

    // If ZenMemPool optimization is enabled(default), update the state of
    // memory pool based on input_array address.
    if ((zen_env_obj.zenEnableMemPool % MEMPOOL_TYPE) &&
        !zendnn_params_.is_eager && zen_pool_buffer) {
      zen_pool_buffer->ZenMemPoolFree(context,
                                      const_cast<float *>(input_array));
    }

    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-DEF: _ZenPool (TF kernel): Compute Is Successful!");
  }

 private:
  std::vector<int32> ksize_, stride_;
  Padding padding_;
  std::vector<int64_t> explicit_paddings_;
  TensorFormat data_format_;
  /* ZenDNN specific */
  ZendnnParameters zendnn_params_;
};

#define REGISTER_POOL_KERNELS(TYPE)                                     \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("_ZenMaxPool").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      ZenPoolOp<TYPE, true>);                                           \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("_ZenAvgPool").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      ZenPoolOp<TYPE, false>);

TF_CALL_float(REGISTER_POOL_KERNELS)

#undef REGISTER_POOL_KERNELS

}  // namespace amd_cpu_plugin
