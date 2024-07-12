/*******************************************************************************
 * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 *******************************************************************************/

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_MATMUL_KERNEL_UTIL_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_MATMUL_KERNEL_UTIL_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow_plugin/src/amd_cpu/util/zen_utils.h"

using zendnn::matmul;
using zendnn::primitive_attr;

namespace amd_cpu_plugin {

struct ZenMatMulParams {
  memory::dims src_dims;
  memory::dims weight_dims;
  memory::dims bias_dims;
  memory::dims dst_dims;
  memory::format_tag src_format;
  memory::format_tag weight_format;
  string dtypes = string("");
  struct PostOpParam {
    string name;
    std::vector<float> param;
  };
  std::vector<PostOpParam> post_op_params;
  bool is_biasadd;

  ZenMatMulParams(memory::dims src_dims, memory::dims weight_dims,
                  memory::dims bias_dims, memory::dims dst_dims,
                  memory::format_tag src_format = memory::format_tag::any,
                  memory::format_tag weight_format = memory::format_tag::any,
                  bool is_biasadd = false)
      : src_dims(src_dims),
        weight_dims(weight_dims),
        bias_dims(bias_dims),
        dst_dims(dst_dims),
        src_format(src_format),
        weight_format(weight_format),
        is_biasadd(is_biasadd) {}
};

template <typename Tinput, typename Tweight, typename Tbias, typename Toutput>
class ZenMatMulPrimitive : public ZenPrimitive {
 public:
  explicit ZenMatMulPrimitive(const ZenMatMulParams &matmul_params)
      : ZenPrimitive() {
    ZenExecutor *ex = ex->getInstance();
    std::shared_ptr<stream> s = ex->getStreamPtr();
    context_.fwd_stream = s;
    // Create matmul primitive.
    if (context_.matmul_fwd == nullptr) {
      Setup(matmul_params);
    }
  }

  ~ZenMatMulPrimitive() {}

  void Execute(const Tinput *src_data, const Tweight *weight_data,
               const Tbias *bias_data, Toutput *dst_data,
               bool is_biasadd = false) {
    // Set data handle.
    context_.src_mem->set_data_handle(
        static_cast<void *>(const_cast<Tinput *>(src_data)));
    context_.weight_mem->set_data_handle(
        static_cast<void *>(const_cast<Tweight *>(weight_data)));
    if (is_biasadd) {
      context_.bias_mem->set_data_handle(
          static_cast<void *>(const_cast<Tbias *>(bias_data)));
    }
    context_.dst_mem->set_data_handle(static_cast<void *>(dst_data));
    // Execute matmul primitive.
    execute_primitives(context_.net, context_.fwd_stream, context_.net_args);
    // Reset data handle back.
    context_.src_mem->set_data_handle(DummyData);
    context_.weight_mem->set_data_handle(DummyData);
    if (is_biasadd) {
      context_.bias_mem->set_data_handle(DummyData);
    }
    context_.dst_mem->set_data_handle(DummyData);
  }

  std::shared_ptr<matmul::primitive_desc> GetPrimitiveDesc() const {
    return context_.matmul_pd;
  }

 private:
  // Primitive reuse context.
  struct ZenMatMulContext {
    // Memory descriptors for matmul source, weights, bias and destination.
    std::shared_ptr<memory::desc> src_md;
    std::shared_ptr<memory::desc> weight_md;
    std::shared_ptr<memory::desc> bias_md;
    std::shared_ptr<memory::desc> dst_md;
    // Memory objects for matmul source, weights, bias and destination.
    std::shared_ptr<memory> src_mem;
    std::shared_ptr<memory> weight_mem;
    std::shared_ptr<memory> bias_mem;
    std::shared_ptr<memory> dst_mem;
    // Operation descriptor.
    std::shared_ptr<matmul::desc> matmul_desc;
    // Primitive descriptor.
    std::shared_ptr<matmul::primitive_desc> matmul_pd;
    // Matmul primitive.
    std::shared_ptr<primitive> matmul_fwd;

    std::shared_ptr<stream> fwd_stream;
    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    ZenMatMulContext()
        : src_md(nullptr),
          weight_md(nullptr),
          bias_md(nullptr),
          dst_md(nullptr),
          src_mem(nullptr),
          weight_mem(nullptr),
          bias_mem(nullptr),
          dst_mem(nullptr),
          matmul_desc(nullptr),
          matmul_pd(nullptr),
          matmul_fwd(nullptr),
          fwd_stream(nullptr) {}
  };

  void Setup(const ZenMatMulParams &matmul_params) {
    bool is_float = std::is_same<Tinput, float>::value;
    // Create memory descriptors.
    auto data_type =
        (is_float) ? memory::data_type::f32 : memory::data_type::bf16;
    context_.src_md.reset(new memory::desc({matmul_params.src_dims}, data_type,
                                           matmul_params.src_format));
    context_.weight_md.reset(new memory::desc(
        {matmul_params.weight_dims}, data_type, matmul_params.weight_format));
    context_.dst_md.reset(new memory::desc({matmul_params.dst_dims}, data_type,
                                           memory::format_tag::nc));
    if (matmul_params.is_biasadd) {
      context_.bias_md.reset(new memory::desc(
          {matmul_params.bias_dims}, data_type, memory::format_tag::nc));
      // Create descriptor for matmul.
      context_.matmul_desc.reset(
          new matmul::desc(*context_.src_md, *context_.weight_md,
                           *context_.bias_md, *context_.dst_md));
    } else {
      context_.matmul_desc.reset(new matmul::desc(
          *context_.src_md, *context_.weight_md, *context_.dst_md));
    }

    // Create primitive descriptor for matmul.
    // Check if there is any fusion as post-ops.
    auto const &post_op_params = matmul_params.post_op_params;
    primitive_attr post_ops_attr;
    post_ops post_ops;
    if (!post_op_params.empty()) {
      for (auto const &post_op_param : post_op_params) {
        if (post_op_param.name == "relu" || post_op_param.name == "sigmoid" ||
            post_op_param.name == "tanh" ||
            post_op_param.name == "GeluApproximate" ||
            post_op_param.name == "GeluExact") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          if (post_op_param.name == "relu") {
            post_ops.append_eltwise(op_scale, algorithm::eltwise_relu, op_alpha,
                                    op_beta);
          } else if (post_op_param.name == "sigmoid") {
            // NOTE: In TensorFlow, tf.nn.sigmoid corresponds to the
            // eltwise_logistic algorithm in OneDNN, which is used here to
            // ensure compatibility with the standard Sigmoid activation (output
            // range 0-1).
            post_ops.append_eltwise(op_scale, algorithm::eltwise_logistic,
                                    op_alpha, op_beta);
          } else if (post_op_param.name == "tanh") {
            post_ops.append_eltwise(op_scale, algorithm::eltwise_tanh, op_alpha,
                                    op_beta);
          } else if (post_op_param.name == "GeluApproximate") {
            post_ops.append_eltwise(op_scale, algorithm::eltwise_gelu, op_alpha,
                                    op_beta);
          } else if (post_op_param.name == "GeluExact") {
            post_ops.append_eltwise(op_scale, algorithm::eltwise_gelu_erf,
                                    op_alpha, op_beta);
          }
        } else if (post_op_param.name == "sum") {
          DCHECK_EQ(post_op_param.param.size(), 1);
          float op_beta = post_op_param.param[0];
          post_ops.append_sum(op_beta);
        } else {
          DCHECK_EQ(post_op_param.name, "relu");
        }
      }
      post_ops_attr.set_post_ops(post_ops);
      context_.matmul_pd.reset(new matmul::primitive_desc(
          *context_.matmul_desc, post_ops_attr, cpu_engine_));
    } else {
      context_.matmul_pd.reset(
          new matmul::primitive_desc(*context_.matmul_desc, cpu_engine_));
    }

    // Create memory primitive based on dummy data.
    context_.src_mem.reset(new memory(context_.matmul_pd.get()->src_desc(),
                                      cpu_engine_, DummyData));
    context_.weight_mem.reset(new memory(
        context_.matmul_pd.get()->weights_desc(), cpu_engine_, DummyData));
    context_.dst_mem.reset(new memory(context_.matmul_pd.get()->dst_desc(),
                                      cpu_engine_, DummyData));
    if (matmul_params.is_biasadd) {
      context_.bias_mem.reset(new memory(context_.matmul_pd.get()->bias_desc(),
                                         cpu_engine_, DummyData));
    }

    // Create primitive for matmul.
    context_.matmul_fwd.reset(new matmul(*context_.matmul_pd));
    if (matmul_params.is_biasadd) {
      context_.net_args.push_back({{ZENDNN_ARG_SRC, *context_.src_mem},
                                   {ZENDNN_ARG_WEIGHTS, *context_.weight_mem},
                                   {ZENDNN_ARG_BIAS, *context_.bias_mem},
                                   {ZENDNN_ARG_DST, *context_.dst_mem}});
    } else {
      context_.net_args.push_back({{ZENDNN_ARG_SRC, *context_.src_mem},
                                   {ZENDNN_ARG_WEIGHTS, *context_.weight_mem},
                                   {ZENDNN_ARG_DST, *context_.dst_mem}});
    }
    context_.net.push_back(*context_.matmul_fwd);
    return;
  }

  struct ZenMatMulContext context_;
};

template <typename Tinput, typename Tweight, typename Tbias, typename Toutput>
class ZenMatMulPrimitiveFactory : public ZenPrimitiveFactory {
 public:
  static ZenMatMulPrimitive<Tinput, Tweight, Tbias, Toutput> *Get(
      const ZenMatMulParams &matmul_dims, bool do_not_cache) {
    ZenMatMulPrimitive<Tinput, Tweight, Tbias, Toutput> *matmul_fwd = nullptr;

    if (do_not_cache) {
      // Always create new primitive.
      matmul_fwd =
          new ZenMatMulPrimitive<Tinput, Tweight, Tbias, Toutput>(matmul_dims);
    } else {
      // Find a suitable existing primitive for a reuse.
      matmul_fwd =
          dynamic_cast<ZenMatMulPrimitive<Tinput, Tweight, Tbias, Toutput> *>(
              ZenMatMulPrimitiveFactory<Tinput, Tweight, Tbias,
                                        Toutput>::GetInstance()
                  .GetMatMul(matmul_dims));
      if (matmul_fwd == nullptr) {
        matmul_fwd = new ZenMatMulPrimitive<Tinput, Tweight, Tbias, Toutput>(
            matmul_dims);
        ZenMatMulPrimitiveFactory<Tinput, Tweight, Tbias,
                                  Toutput>::GetInstance()
            .SetMatMul(matmul_dims, matmul_fwd);
      }
    }
    return matmul_fwd;
  }

 private:
  ZenMatMulPrimitiveFactory() {}
  ~ZenMatMulPrimitiveFactory() {}

  static ZenMatMulPrimitiveFactory &GetInstance() {
    static ZenMatMulPrimitiveFactory instance_;
    return instance_;
  }

  static string CreateKey(const ZenMatMulParams &matmul_dims) {
    string prefix = "matmul_fwd_";
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(matmul_dims.src_dims);
    key_creator.AddAsKey(matmul_dims.weight_dims);
    key_creator.AddAsKey(matmul_dims.bias_dims);
    key_creator.AddAsKey(matmul_dims.dst_dims);
    key_creator.AddAsKey(matmul_dims.dtypes);

    // Generate keys for post-ops.
    for (auto const &post_op_param : matmul_dims.post_op_params) {
      if (post_op_param.name == "relu" || post_op_param.name == "sigmoid" ||
          post_op_param.name == "GeluApproximate" ||
          post_op_param.name == "tanh" || post_op_param.name == "GeluExact") {
        DCHECK_EQ(post_op_param.param.size(), 3);
        key_creator.AddAsKey(post_op_param.name);
        key_creator.AddAsKey(post_op_param.param[0]);
        key_creator.AddAsKey(post_op_param.param[1]);
        key_creator.AddAsKey(post_op_param.param[2]);
      } else if (post_op_param.name == "sum") {
        DCHECK_EQ(post_op_param.param.size(), 1);
        key_creator.AddAsKey(post_op_param.name);
        key_creator.AddAsKey(post_op_param.param[0]);
      } else {
        return string("not_a_key");
      }
    }
    return key_creator.GetKey();
  }

  ZenPrimitive *GetMatMul(const ZenMatMulParams &matmul_dims) {
    string key = CreateKey(matmul_dims);
    return this->GetOp(key);
  }

  void SetMatMul(const ZenMatMulParams &matmul_dims, ZenPrimitive *op) {
    string key = CreateKey(matmul_dims);
    this->SetOp(key, op);
  }
};

}  // namespace amd_cpu_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_MATMUL_KERNEL_UTIL_H_
