/*******************************************************************************
 * Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 ******************************************************************************/

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_CONV_KERNEL_UTIL_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_CONV_KERNEL_UTIL_H_

// Standard headers
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
// TensorFlow plug-in headers
#include "tensorflow_plugin/src/amd_cpu/util/zen_utils.h"

using zendnn::convolution_forward;
using zendnn::primitive_attr;
using zendnn::prop_kind;
using zendnn::stream;
using std::string;

namespace amd_cpu_plugin {

void ZenConvolution2DDepthwise(
    zendnn::engine eng, zendnn::stream s, zendnn::primitive_attr conv_attr,
    void *input_array, int batch_size, int channels, int height, int width,
    void *filter_array, int output_channels, int kernel_h, int kernel_w,
    float pad_t, float pad_l, float pad_b, float pad_r, int stride_h,
    int stride_w, void *bias_array, void *output_array, int out_height,
    int out_width, bool is_eager, bool reorder_before, bool reorder_after,
    void *cached_filter_data_ptensor_, void *context);

struct ZenConvParams {
  memory::dims src_dims;
  memory::dims filter_dims;
  memory::dims bias_dims;
  memory::dims dst_dims;
  memory::dims strides;
  memory::dims padding_left;
  memory::dims padding_right;
  memory::data_type src_dt;
  memory::data_type filter_dt;
  memory::data_type bias_dt;
  memory::data_type dst_dt;
  memory::format_tag weight_tag;
  void *cached_ptr;
  zendnn::algorithm conv_algo;
  std::vector<float> conv_scale;
  string dtypes = string("");
  struct PostOpParam {
    string name;
    std::vector<float> param;
  };
  std::vector<PostOpParam> post_op_params;

  ZenConvParams(memory::dims src_dims, memory::dims filter_dims,
                memory::dims bias_dims, memory::dims dst_dims,
                memory::dims strides, memory::dims padding_left,
                memory::dims padding_right, memory::data_type src_dt,
                memory::data_type filter_dt, memory::data_type bias_dt,
                memory::data_type dst_dt, memory::format_tag weight_tag,
                void *ptr, zendnn::algorithm algo, std::vector<float> scale)
      : src_dims(src_dims),
        filter_dims(filter_dims),
        bias_dims(bias_dims),
        dst_dims(dst_dims),
        strides(strides),
        padding_left(padding_left),
        padding_right(padding_right),
        src_dt(src_dt),
        filter_dt(filter_dt),
        bias_dt(bias_dt),
        dst_dt(dst_dt),
        weight_tag(weight_tag),
        cached_ptr(ptr),
        conv_algo(algo),
        conv_scale(scale) {}
};

class ZenConvPrimitive : public ZenPrimitive {
 public:
  explicit ZenConvPrimitive(const ZenConvParams &conv_params) : ZenPrimitive() {
    ZenExecutor *ex = ex->getInstance();
    std::shared_ptr<stream> s = ex->getStreamPtr();
    context_.conv_stream = s;
    // Create convolution primitive.
    if (context_.conv_prim == nullptr) {
      Setup(conv_params);
    }
  }

  ~ZenConvPrimitive() {}

  void Execute(const float *src_data, const float *filter_data,
               const float *bias_data, float *dst_data) {
    // Set data handle.
    context_.src_mem->set_data_handle(
        static_cast<void *>(const_cast<float *>(src_data)));
    context_.filter_mem->set_data_handle(
        static_cast<void *>(const_cast<float *>(filter_data)));
    context_.bias_mem->set_data_handle(
        static_cast<void *>(const_cast<float *>(bias_data)));
    context_.dst_mem->set_data_handle(
        static_cast<void *>(static_cast<float *>(dst_data)));
    // Execute matmul primitive.
    execute_primitives(context_.net, context_.conv_stream, context_.net_args);
    // Reset data handle back.
    context_.src_mem->set_data_handle(DummyData);
    context_.filter_mem->set_data_handle(DummyData);
    context_.bias_mem->set_data_handle(DummyData);
    context_.dst_mem->set_data_handle(DummyData);
  }

  void Execute(const float *src_data, const float *filter_data,
               const float *bias_data, float *dst_data,
               const float *bn_scale_data, const float *bn_mean_data,
               const float *bn_offset_data) {
    // Set data handle.
    context_.src_mem->set_data_handle(
        static_cast<void *>(const_cast<float *>(src_data)));
    context_.filter_mem->set_data_handle(
        static_cast<void *>(const_cast<float *>(filter_data)));
    context_.bias_mem->set_data_handle(
        static_cast<void *>(const_cast<float *>(bias_data)));
    context_.dst_mem->set_data_handle(
        static_cast<void *>(static_cast<float *>(dst_data)));
    context_.bn_scale_mem->set_data_handle(
        static_cast<void *>(const_cast<float *>(bn_scale_data)));
    context_.bn_mean_mem->set_data_handle(
        static_cast<void *>(const_cast<float *>(bn_mean_data)));
    context_.bn_offset_mem->set_data_handle(
        static_cast<void *>(const_cast<float *>(bn_offset_data)));
    // Execute matmul primitive.
    execute_primitives(context_.net, context_.conv_stream, context_.net_args);
    // Reset data handle back.
    context_.src_mem->set_data_handle(DummyData);
    context_.filter_mem->set_data_handle(DummyData);
    context_.bias_mem->set_data_handle(DummyData);
    context_.dst_mem->set_data_handle(DummyData);
    context_.bn_scale_mem->set_data_handle(DummyData);
    context_.bn_mean_mem->set_data_handle(DummyData);
    context_.bn_offset_mem->set_data_handle(DummyData);
  }

  std::shared_ptr<convolution_forward::primitive_desc> GetPrimitiveDesc()
      const {
    return context_.conv_pd;
  }

 private:
  // Primitive reuse context.
  struct ZenConvContext {
    // Memory descriptors for matmul source, weights, bias and destination.
    std::shared_ptr<memory::desc> src_md;
    std::shared_ptr<memory::desc> filter_md;
    std::shared_ptr<memory::desc> bias_md;
    std::shared_ptr<memory::desc> dst_md;
    // Memory descriptors for batchnorm fusion.
    std::shared_ptr<memory::desc> bn_scale_md;
    std::shared_ptr<memory::desc> bn_mean_md;
    std::shared_ptr<memory::desc> bn_offset_md;
    // Memory objects for matmul source, weights, bias and destination.
    std::shared_ptr<memory> src_mem;
    std::shared_ptr<memory> filter_mem;
    std::shared_ptr<memory> bias_mem;
    std::shared_ptr<memory> dst_mem;
    // Memory objects for batchnorm fusion.
    std::shared_ptr<memory> bn_scale_mem;
    std::shared_ptr<memory> bn_mean_mem;
    std::shared_ptr<memory> bn_offset_mem;

    // Operation descriptor.
    std::shared_ptr<convolution_forward::desc> conv_desc;
    // Primitive descriptor.
    std::shared_ptr<convolution_forward::primitive_desc> conv_pd;
    // Convolution primitive.
    std::shared_ptr<primitive> conv_prim;

    std::shared_ptr<stream> conv_stream;
    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    ZenConvContext()
        : src_md(nullptr),
          filter_md(nullptr),
          bias_md(nullptr),
          dst_md(nullptr),
          bn_scale_md(nullptr),
          bn_mean_md(nullptr),
          bn_offset_md(nullptr),
          src_mem(nullptr),
          filter_mem(nullptr),
          bias_mem(nullptr),
          dst_mem(nullptr),
          bn_scale_mem(nullptr),
          bn_mean_mem(nullptr),
          bn_offset_mem(nullptr),
          conv_desc(nullptr),
          conv_pd(nullptr),
          conv_prim(nullptr),
          conv_stream(nullptr) {}
  };

  void Setup(const ZenConvParams &conv_params) {
    // Create memory descriptors.
    context_.src_md.reset(new memory::desc(
        {conv_params.src_dims}, conv_params.src_dt, memory::format_tag::nhwc));
    // memory::format_tag::acdb));
    context_.filter_md.reset(new memory::desc({conv_params.filter_dims},
                                              conv_params.filter_dt,
                                              memory::format_tag::any));
    context_.dst_md.reset(new memory::desc(
        {conv_params.dst_dims}, conv_params.dst_dt, memory::format_tag::nhwc));
    // memory::format_tag::acdb));
    context_.bias_md.reset(new memory::desc(
        {conv_params.bias_dims}, conv_params.bias_dt, memory::format_tag::x));
    // Create operation descriptor.
    context_.conv_desc.reset(new convolution_forward::desc(
        prop_kind::forward_inference, conv_params.conv_algo, *context_.src_md,
        *context_.filter_md, *context_.bias_md, *context_.dst_md,
        conv_params.strides, conv_params.padding_left,
        conv_params.padding_right));
    // Create primitive descriptor.
    context_.conv_pd.reset(new convolution_forward::primitive_desc(
        *context_.conv_desc, cpu_engine_));

    // If there are post-ops, then update operation and primitive descriptor.
    bool fused_batchnorm = false;
    bool fused_relu = false;
    auto const &post_op_params = conv_params.post_op_params;
    primitive_attr post_ops_attr;
    zendnn::post_ops post_ops;
    if (!post_op_params.empty()) {
      for (auto const &post_op_param : post_op_params) {
        if (post_op_param.name == "relu") {
          fused_relu = true;
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          post_ops.append_eltwise(op_scale, zendnn::algorithm::eltwise_relu,
                                  op_alpha, op_beta);
        } else if (post_op_param.name == "sum") {
          DCHECK_EQ(post_op_param.param.size(), 1);
          float op_scale = post_op_param.param[0];
          post_ops.append_sum(op_scale);
        } else if (post_op_param.name == "batchnorm") {
          fused_batchnorm = true;
        } else if (post_op_param.name == "scale") {
          DCHECK_EQ(post_op_param.param.size(), 1);
          float op_scale = post_op_param.param[0];
          post_ops_attr.set_output_scales(op_scale, conv_params.conv_scale);
        } else {
          DCHECK((post_op_param.name == "relu") ||
                 (post_op_param.name == "sum") ||
                 (post_op_param.name == "batchnorm"));
        }
      }
      // Update operation descriptor for fused batchnorm.
      if (fused_batchnorm) {
        context_.bn_scale_md.reset(new memory::desc({conv_params.bias_dims},
                                                    memory::data_type::f32,
                                                    memory::format_tag::x));
        context_.bn_mean_md.reset(new memory::desc({conv_params.bias_dims},
                                                   memory::data_type::f32,
                                                   memory::format_tag::x));
        context_.bn_offset_md.reset(new memory::desc({conv_params.bias_dims},
                                                     memory::data_type::f32,
                                                     memory::format_tag::x));
        context_.conv_desc.reset(new convolution_forward::desc(
            prop_kind::forward_inference, conv_params.conv_algo,
            *context_.src_md, *context_.filter_md, *context_.bias_md,
            *context_.dst_md, conv_params.strides, conv_params.padding_left,
            conv_params.padding_right, fused_relu, fused_batchnorm,
            *context_.bn_scale_md, *context_.bn_mean_md,
            *context_.bn_offset_md));
      }
      post_ops_attr.set_post_ops(post_ops);
      context_.conv_pd.reset(new convolution_forward::primitive_desc(
          *context_.conv_desc, post_ops_attr, cpu_engine_));
    }

    // Create memory primitive based on dummy data.
    context_.src_mem.reset(
        new memory(context_.conv_pd.get()->src_desc(), cpu_engine_, DummyData));
    context_.filter_mem.reset(new memory(context_.conv_pd.get()->weights_desc(),
                                         cpu_engine_, DummyData));
    context_.dst_mem.reset(
        new memory(context_.conv_pd.get()->dst_desc(), cpu_engine_, DummyData));
    context_.bias_mem.reset(new memory(context_.conv_pd.get()->bias_desc(),
                                       cpu_engine_, DummyData));
    if (!fused_batchnorm) {
      context_.net_args.push_back({{ZENDNN_ARG_SRC, *context_.src_mem},
                                   {ZENDNN_ARG_WEIGHTS, *context_.filter_mem},
                                   {ZENDNN_ARG_BIAS, *context_.bias_mem},
                                   {ZENDNN_ARG_DST, *context_.dst_mem}});
    } else {
      // Update memory descriptor for fused batchnorm with dummy data.
      context_.bn_scale_mem.reset(
          new memory(*context_.bn_scale_md, cpu_engine_, DummyData));
      context_.bn_mean_mem.reset(
          new memory(*context_.bn_mean_md, cpu_engine_, DummyData));
      context_.bn_offset_mem.reset(
          new memory(*context_.bn_offset_md, cpu_engine_, DummyData));
      context_.net_args.push_back(
          {{ZENDNN_ARG_SRC, *context_.src_mem},
           {ZENDNN_ARG_WEIGHTS, *context_.filter_mem},
           {ZENDNN_ARG_BIAS, *context_.bias_mem},
           {ZENDNN_ARG_DST, *context_.dst_mem},
           {ZENDNN_ARG_BN_SCALE, *context_.bn_scale_mem},
           {ZENDNN_ARG_BN_MEAN, *context_.bn_mean_mem},
           {ZENDNN_ARG_BN_OFFSET, *context_.bn_offset_mem}});
    }

    // Create primitive for convolution.
    context_.conv_prim.reset(new convolution_forward(*context_.conv_pd));
    context_.net.push_back(*context_.conv_prim);
  }

  struct ZenConvContext context_;
};

class ZenConvPrimitiveFactory : public ZenPrimitiveFactory {
 public:
  static ZenConvPrimitive *Get(const ZenConvParams &conv_params,
                               bool reuse_opt_disabled) {
    ZenConvPrimitive *conv_prim = nullptr;

    if (reuse_opt_disabled) {
      // Primitive reuse optimization is disabled hence create a new primitive.
      conv_prim = new ZenConvPrimitive(conv_params);
    } else {
      // Find a suitable existing primitive for a reuse.
      conv_prim = dynamic_cast<ZenConvPrimitive *>(
          ZenConvPrimitiveFactory::GetInstance().GetConvPrimitive(conv_params));
      if (conv_prim == nullptr) {
        conv_prim = new ZenConvPrimitive(conv_params);
        ZenConvPrimitiveFactory::GetInstance().SetConvPrimitive(conv_params,
                                                                conv_prim);
      }
    }
    return conv_prim;
  }

 private:
  ZenConvPrimitiveFactory() {}
  ~ZenConvPrimitiveFactory() {}

  static const int kDilationH = 0, kDilationW = 1;

  static ZenConvPrimitiveFactory &GetInstance() {
    static ZenConvPrimitiveFactory instance_;
    return instance_;
  }

  static string CreateKey(const ZenConvParams &conv_params) {
    string prefix = "conv_prim_";
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(conv_params.cached_ptr);
    key_creator.AddAsKey(conv_params.src_dims);
    key_creator.AddAsKey(conv_params.filter_dims);
    key_creator.AddAsKey(conv_params.bias_dims);
    key_creator.AddAsKey(conv_params.dst_dims);
    key_creator.AddAsKey(conv_params.strides);
    key_creator.AddAsKey(conv_params.padding_left);
    key_creator.AddAsKey(conv_params.padding_right);
    key_creator.AddAsKey(conv_params.dtypes);

    // Generate keys for post-ops.
    for (auto const &post_op_param : conv_params.post_op_params) {
      if (post_op_param.name == "relu") {
        DCHECK_EQ(post_op_param.param.size(), 3);
      } else if (post_op_param.name == "sum") {
        DCHECK_EQ(post_op_param.param.size(), 1);
      } else if (post_op_param.name == "scale") {
        DCHECK_EQ(post_op_param.param.size(), 1);
      } else if (post_op_param.name != "batchnorm") {
        return string("not_a_key");
      }
      key_creator.AddAsKey(post_op_param.name);
      for (auto &param : post_op_param.param) {
        key_creator.AddAsKey(param);
      }
    }
    return key_creator.GetKey();
  }

  ZenPrimitive *GetConvPrimitive(const ZenConvParams &conv_params) {
    string key = CreateKey(conv_params);
    return this->GetOp(key);
  }

  void SetConvPrimitive(const ZenConvParams &conv_params, ZenPrimitive *op) {
    string key = CreateKey(conv_params);
    this->SetOp(key, op);
  }
};

}  // namespace amd_cpu_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_CONV_KERNEL_UTIL_H__
