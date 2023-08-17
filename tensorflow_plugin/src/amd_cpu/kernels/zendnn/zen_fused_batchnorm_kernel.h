/*******************************************************************************
 * Modifications Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 ******************************************************************************/

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_FUSED_BATCHNORM_KERNEL_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_FUSED_BATCHNORM_KERNEL_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// TensorFlow plug-in headers.
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/fused_eigen_output_kernels.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_kernel_common.h"
#include "tensorflow_plugin/src/amd_cpu/util/bounds_check.h"
#include "tensorflow_plugin/src/amd_cpu/util/common_shape_fns.h"
#include "tensorflow_plugin/src/amd_cpu/util/errors.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_kernel.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_requires.h"
#include "tensorflow_plugin/src/amd_cpu/util/padding.h"
#include "tensorflow_plugin/src/amd_cpu/util/tensor_format.h"
#include "tensorflow_plugin/src/amd_cpu/util/zen_utils.h"

#define GET_FLAG(bn_flag) static_cast<int>(zendnn::normalization_flags::bn_flag)
#define IS_SET(cflag) (context_.flags & GET_FLAG(cflag))

using zendnn::batch_normalization_forward;
using zendnn::prop_kind;
using zendnn::stream;
using BatchNormFwdPd = batch_normalization_forward::primitive_desc;
using CPUDevice = Eigen::ThreadPoolDevice;

namespace amd_cpu_plugin {

namespace functor {

// FusedBatchNormEx op supports side inputs and activations:
// (1) batch_norm + activation
// (2) batch norm + side input + activation
enum class FusedBatchNormActivationMode { kIdentity, kRelu };

string ToString(FusedBatchNormActivationMode activation_mode) {
  switch (activation_mode) {
    case FusedBatchNormActivationMode::kIdentity:
      return "Identity";
    case FusedBatchNormActivationMode::kRelu:
      return "Relu";
  }
}

}  // namespace functor

using FusedBNActivationMode = functor::FusedBatchNormActivationMode;

Status ParseActivationMode(OpKernelConstruction *context,
                           FusedBNActivationMode *activation_mode) {
  string activation_mode_str;
  context->GetAttr("activation_mode", &activation_mode_str);

  if (activation_mode_str == "Identity") {
    *activation_mode = FusedBNActivationMode::kIdentity;
    return Status::OK();
  }
  if (activation_mode_str == "Relu") {
    *activation_mode = FusedBNActivationMode::kRelu;
    return Status::OK();
  }
  return errors::InvalidArgument("Unsupported activation mode: ",
                                 activation_mode_str);
}

struct ZenBatchNormFwdParams {
  memory::dims src_dims;
  int depth;
  float eps;
  bool training;
  FusedBNActivationMode activation_mode;
  memory::desc src_md;

  ZenBatchNormFwdParams(const memory::dims &src_dims, int depth, float eps,
                        bool training, memory::desc src_md,
                        FusedBNActivationMode activation_mode)
      : src_dims(src_dims),
        depth(depth),
        eps(eps),
        training(training),
        activation_mode(activation_mode),
        src_md(src_md) {}
};

template <typename T, typename U>
class ZenFusedBatchNormFwdPrimitive : public ZenPrimitive {
 public:
  explicit ZenFusedBatchNormFwdPrimitive(const ZenBatchNormFwdParams &fwdParams)
      : ZenPrimitive() {
    ZenExecutor *ex = ex->getInstance();
    std::shared_ptr<stream> s = ex->getStreamPtr();
    context_.bn_stream = s;
    if (context_.bn_fwd == nullptr) {
      Setup(fwdParams);
    }
  }

  ~ZenFusedBatchNormFwdPrimitive() {}

  // BatchNormalization forward execute
  // @input src_data -input data buffer of src,
  //        weights_data - input data buffer of weights,
  //        dst_data - output data buffer of dst,
  //        mean_data - output data buffer of means,
  //        variance_data - output data buffer of variances.
  void Execute(const T *src_data, const U *weights_data, T *dst_data,
               U *mean_data, U *variance_data, U *workspace_data) {
    context_.src_mem->set_data_handle(
        static_cast<void *>(const_cast<T *>(src_data)));
    context_.dst_mem->set_data_handle(static_cast<void *>(dst_data));

    if (IS_SET(use_scale_shift))
      context_.weights_mem->set_data_handle(
          static_cast<void *>(const_cast<U *>(weights_data)));

    if ((context_.pkind == prop_kind::forward_training) ||
        (IS_SET(use_global_stats))) {
      context_.mean_mem->set_data_handle(static_cast<void *>(mean_data));
      context_.variance_mem->set_data_handle(
          static_cast<void *>(variance_data));
    }
    if (workspace_data != nullptr) {
      context_.ws_mem->set_data_handle(workspace_data);
    }

    // Execute batch-normalization forward primitives.
    execute_primitives(context_.fwd_primitives, context_.bn_stream,
                       context_.net_args);

    context_.src_mem->set_data_handle(DummyData);
    context_.dst_mem->set_data_handle(DummyData);

    if (IS_SET(use_scale_shift)) {
      context_.weights_mem->set_data_handle(DummyData);
    }

    if ((context_.pkind == prop_kind::forward_training) ||
        (IS_SET(use_global_stats))) {
      context_.mean_mem->set_data_handle(DummyData);
      context_.variance_mem->set_data_handle(DummyData);
    }

    if (workspace_data != nullptr) {
      context_.ws_mem->set_data_handle(DummyData);
    }
  }

  memory::desc GetDstPd() const { return context_.dst_mem->get_desc(); }

  std::shared_ptr<BatchNormFwdPd> GetBatchNormFwdPd() const {
    return context_.fwd_pd;
  }

 private:
  // Primitive reuse context for BatchNorm forward op.
  struct BatchNormFwdContext {
    // Flags indicating if it is training or inference mode.
    int64 flags;

    // Algorithm kind.
    prop_kind pkind;

    // Inputs/outputs memory.
    std::shared_ptr<memory> src_mem;
    std::shared_ptr<memory> weights_mem;
    std::shared_ptr<memory> dst_mem;
    std::shared_ptr<memory> mean_mem;
    std::shared_ptr<memory> variance_mem;
    std::shared_ptr<memory> ws_mem;

    // Forward BatchNorm primitive descriptor.
    std::shared_ptr<BatchNormFwdPd> fwd_pd;

    // BatchNorm forward primitive.
    std::shared_ptr<primitive> bn_fwd;
    std::vector<primitive> fwd_primitives;

    std::vector<std::unordered_map<int, memory>> net_args;

    std::shared_ptr<stream> bn_stream;

    BatchNormFwdContext()
        : flags(0),
          pkind(prop_kind::forward_training),
          src_mem(nullptr),
          weights_mem(nullptr),
          dst_mem(nullptr),
          mean_mem(nullptr),
          variance_mem(nullptr),
          ws_mem(nullptr),
          bn_fwd(nullptr),
          bn_stream(nullptr) {}
  };

  void Setup(const ZenBatchNormFwdParams &fwdParams) {
    context_.flags =
        fwdParams.training
            ? GET_FLAG(use_scale_shift)
            : (GET_FLAG(use_scale_shift) | GET_FLAG(use_global_stats));
    context_.pkind = fwdParams.training ? prop_kind::forward_training
                                        : prop_kind::forward_scoring;

    if (fwdParams.activation_mode == FusedBNActivationMode::kRelu) {
      context_.flags |= GET_FLAG(fuse_norm_relu);
    }
    // Memory descriptor.
    auto src_md = fwdParams.src_md;
    // Create forward BatchNorm descriptor and primitive descriptor.
    auto fwd_desc = batch_normalization_forward::desc(
        context_.pkind, src_md, fwdParams.eps,
        static_cast<zendnn::normalization_flags>(context_.flags));

    context_.fwd_pd.reset(new BatchNormFwdPd(fwd_desc, cpu_engine_));

    // Create memory primitive based on dummy data.
    context_.src_mem.reset(
        new memory(context_.fwd_pd->src_desc(), cpu_engine_, DummyData));
    context_.dst_mem.reset(
        new memory(context_.fwd_pd->dst_desc(), cpu_engine_, DummyData));

    memory::dims s_dims = {2, fwdParams.depth};
    memory::dims m_dims = {1, fwdParams.depth};
    if (IS_SET(use_scale_shift)) {
      context_.weights_mem.reset(
          new memory({{s_dims}, memory::data_type::f32, memory::format_tag::nc},
                     cpu_engine_, DummyData));
    }

    if (fwdParams.training || (IS_SET(use_global_stats))) {
      context_.mean_mem.reset(
          new memory({{m_dims}, memory::data_type::f32, memory::format_tag::nc},
                     cpu_engine_, DummyData));

      context_.variance_mem.reset(
          new memory({{m_dims}, memory::data_type::f32, memory::format_tag::nc},
                     cpu_engine_, DummyData));
    }

    if (IS_SET(fuse_norm_relu)) {
      context_.ws_mem.reset(new memory(context_.fwd_pd->workspace_desc(),
                                       cpu_engine_, DummyData));
    }

    // BatchNorm forward primitive.
    if (!fwdParams.training && !(IS_SET(use_global_stats))) {
      if ((IS_SET(use_scale_shift)) && zendnn_use_scaleshift) {
        context_.net_args.push_back(
            {{ZENDNN_ARG_SRC, *context_.src_mem},
             {ZENDNN_ARG_WEIGHTS, *context_.weights_mem},
             {ZENDNN_ARG_DST, *context_.dst_mem}});
      } else {
        context_.net_args.push_back({{ZENDNN_ARG_SRC, *context_.src_mem},
                                     {ZENDNN_ARG_DST, *context_.dst_mem}});
      }
      context_.bn_fwd.reset(new batch_normalization_forward(*context_.fwd_pd));
    } else if (IS_SET(use_global_stats)) {
      if ((IS_SET(use_scale_shift)) && GET_FLAG(use_scale_shift)) {
        if (IS_SET(fuse_norm_relu)) {
          context_.net_args.push_back(
              {{ZENDNN_ARG_SRC, *context_.src_mem},
               {ZENDNN_ARG_MEAN, *context_.mean_mem},
               {ZENDNN_ARG_VARIANCE, *context_.variance_mem},
               {ZENDNN_ARG_WEIGHTS, *context_.weights_mem},
               {ZENDNN_ARG_DST, *context_.dst_mem},
               {ZENDNN_ARG_WORKSPACE, *context_.ws_mem}});
        } else {
          context_.net_args.push_back(
              {{ZENDNN_ARG_SRC, *context_.src_mem},
               {ZENDNN_ARG_MEAN, *context_.mean_mem},
               {ZENDNN_ARG_VARIANCE, *context_.variance_mem},
               {ZENDNN_ARG_WEIGHTS, *context_.weights_mem},
               {ZENDNN_ARG_DST, *context_.dst_mem}});
        }
      } else {
        if (IS_SET(fuse_norm_relu)) {
          context_.net_args.push_back(
              {{ZENDNN_ARG_SRC, *context_.src_mem},
               {ZENDNN_ARG_MEAN, *context_.mean_mem},
               {ZENDNN_ARG_VARIANCE, *context_.variance_mem},
               {ZENDNN_ARG_DST, *context_.dst_mem},
               {ZENDNN_ARG_WORKSPACE, *context_.ws_mem}});
        } else {
          context_.net_args.push_back(
              {{ZENDNN_ARG_SRC, *context_.src_mem},
               {ZENDNN_ARG_MEAN, *context_.mean_mem},
               {ZENDNN_ARG_VARIANCE, *context_.variance_mem},
               {ZENDNN_ARG_DST, *context_.dst_mem}});
        }
      }
      context_.bn_fwd.reset(new batch_normalization_forward(*context_.fwd_pd));
    } else {
      if ((IS_SET(use_scale_shift)) && GET_FLAG(use_scale_shift)) {
        if (IS_SET(fuse_norm_relu)) {
          context_.net_args.push_back(
              {{ZENDNN_ARG_SRC, *context_.src_mem},
               {ZENDNN_ARG_WEIGHTS, *context_.weights_mem},
               {ZENDNN_ARG_DST, *context_.dst_mem},
               {ZENDNN_ARG_MEAN, *context_.mean_mem},
               {ZENDNN_ARG_VARIANCE, *context_.variance_mem},
               {ZENDNN_ARG_WORKSPACE, *context_.ws_mem}});
        } else {
          context_.net_args.push_back(
              {{ZENDNN_ARG_SRC, *context_.src_mem},
               {ZENDNN_ARG_WEIGHTS, *context_.weights_mem},
               {ZENDNN_ARG_DST, *context_.dst_mem},
               {ZENDNN_ARG_MEAN, *context_.mean_mem},
               {ZENDNN_ARG_VARIANCE, *context_.variance_mem}});
        }
      } else {
        if (IS_SET(fuse_norm_relu)) {
          context_.net_args.push_back(
              {{ZENDNN_ARG_SRC, *context_.src_mem},
               {ZENDNN_ARG_DST, *context_.dst_mem},
               {ZENDNN_ARG_MEAN, *context_.mean_mem},
               {ZENDNN_ARG_VARIANCE, *context_.variance_mem},
               {ZENDNN_ARG_WORKSPACE, *context_.ws_mem}});
        } else {
          context_.net_args.push_back(
              {{ZENDNN_ARG_SRC, *context_.src_mem},
               {ZENDNN_ARG_DST, *context_.dst_mem},
               {ZENDNN_ARG_MEAN, *context_.mean_mem},
               {ZENDNN_ARG_VARIANCE, *context_.variance_mem}});
        }
      }
      context_.bn_fwd.reset(new batch_normalization_forward(*context_.fwd_pd));
    }

    context_.fwd_primitives.push_back(*context_.bn_fwd);
  }

  struct BatchNormFwdContext context_;
};

template <typename T, typename U>
class ZenFusedBatchNormFwdPrimitiveFactory : public ZenPrimitiveFactory {
 public:
  static ZenFusedBatchNormFwdPrimitive<T, U> *Get(
      const ZenBatchNormFwdParams &fwdParams) {
    auto bn_fwd = static_cast<ZenFusedBatchNormFwdPrimitive<T, U> *>(
        ZenFusedBatchNormFwdPrimitiveFactory<T, U>::GetInstance()
            .GetBatchNormFwd(fwdParams));

    if (bn_fwd == nullptr) {
      bn_fwd = new ZenFusedBatchNormFwdPrimitive<T, U>(fwdParams);
      ZenFusedBatchNormFwdPrimitiveFactory<T, U>::GetInstance().SetBatchNormFwd(
          fwdParams, bn_fwd);
    }
    return bn_fwd;
  }

  static ZenFusedBatchNormFwdPrimitiveFactory &GetInstance() {
    static ZenFusedBatchNormFwdPrimitiveFactory instance_;
    return instance_;
  }

 private:
  ZenFusedBatchNormFwdPrimitiveFactory() {}
  ~ZenFusedBatchNormFwdPrimitiveFactory() {}

  static string CreateKey(const ZenBatchNormFwdParams &fwdParams) {
    string prefix = "bn_fwd";
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(fwdParams.src_dims);
    key_creator.AddAsKey<int>(fwdParams.depth);
    key_creator.AddAsKey<float>(fwdParams.eps);
    key_creator.AddAsKey<bool>(fwdParams.training);
    key_creator.AddAsKey<FusedBNActivationMode>(fwdParams.activation_mode);
    key_creator.AddAsKey(typeid(T).name());
    key_creator.AddAsKey(typeid(U).name());
    return key_creator.GetKey();
  }

  ZenPrimitive *GetBatchNormFwd(const ZenBatchNormFwdParams &fwdParams) {
    string key = CreateKey(fwdParams);
    return this->GetOp(key);
  }

  void SetBatchNormFwd(const ZenBatchNormFwdParams &fwdParams,
                       ZenPrimitive *op) {
    string key = CreateKey(fwdParams);
    this->SetOp(key, op);
  }
};

}  // namespace amd_cpu_plugin

#undef GET_FLAG
#undef IS_SET

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_FUSED_BATCHNORM_KERNEL_H_
