/*******************************************************************************
 * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights
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

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_GRAPH_AUTO_MIXED_PRECISION_AUTO_MIXED_PRECISION_LISTS_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_GRAPH_AUTO_MIXED_PRECISION_AUTO_MIXED_PRECISION_LISTs_H_

#include <string>

#include "tensorflow_plugin/src/amd_cpu/graph/config_util.h"
#include "tensorflow_plugin/src/amd_cpu/util/env_var.h"
#include "tensorflow_plugin/src/amd_cpu/util/gtl/flatset.h"
#include "tensorflow_plugin/src/amd_cpu/util/protobuf/config.pb.h"
#include "tensorflow_plugin/src/amd_cpu/util/status.h"
#include "tensorflow_plugin/src/amd_cpu/util/str_util.h"

namespace amd_cpu_plugin {
namespace graph {

// Represents the four lists of ops: the allow list, infer list, deny list, and
// clear list. These lists determine which ops are converted to bf16
// (referred to as 'f16' for short) and which ops stay as fp32.
class AutoMixedPrecisionLists {
 public:
  virtual ~AutoMixedPrecisionLists() {}

  // Returns the set of ops that are considered numerically-safe (for execution
  // in f16), performance-critical, and can run in f16. These ops are always
  // converted to f16.
  virtual gtl::FlatSet<string> AllowList() = 0;
  // Returns the set of ops that can run in f16 and are considered numerically-
  // safe (for execution in f16), but which may be made unsafe by an upstream
  // denylist op.
  virtual gtl::FlatSet<string> InferList() = 0;
  // Returns the set of ops that are considered numerically-dangerous (i.e.,
  // unsafe for execution in f16) and whose effects may also be observed in
  // downstream nodes (e.g. for f16, in Exp -> Add, the Add is unsafe due to
  // the Exp).
  virtual gtl::FlatSet<string> DenyList() = 0;
  // Returns the set of ops that do not have numerically-significant effects
  // (i.e., they are always considered safe for execution in f16 precision), and
  // can run in f16.
  virtual gtl::FlatSet<string> ClearList() = 0;

 protected:
  // Adds or removes ops from list if certain environmental variables are set.

  static void UpdateList(const string& list_name, gtl::FlatSet<string>* list) {
    CHECK(list_name == "ALLOWLIST" || list_name == "INFERLIST" ||  // Crash OK.
          list_name == "DENYLIST" || list_name == "CLEARLIST");
    string add_env_var = "ZEN_AUTO_MIXED_PRECISION_" + list_name + "_ADD";
    string remove_env_var = "ZEN_AUTO_MIXED_PRECISION_" + list_name + "_REMOVE";
    string to_add, to_remove;

    auto cfg_ = amd_cpu_plugin::zen_get_config();

#define LIST_IS_NOT_NULL(list) \
  cfg_.graph_options().auto_mixed_precision_options().list()

    if (list_name == "ALLOWLIST") {
      to_add = LIST_IS_NOT_NULL(allowlist_add);
      to_remove = LIST_IS_NOT_NULL(allowlist_remove);
      if (to_add.empty()) {
        TF_CHECK_OK(ReadStringFromEnvVar(add_env_var, "", &to_add));
      }
      if (to_remove.empty()) {
        TF_CHECK_OK(ReadStringFromEnvVar(remove_env_var, "", &to_remove));
      }
    }
    if (list_name == "INFERLIST") {
      to_add = LIST_IS_NOT_NULL(inferlist_add);
      to_remove = LIST_IS_NOT_NULL(inferlist_remove);
      if (to_add.empty()) {
        TF_CHECK_OK(ReadStringFromEnvVar(add_env_var, "", &to_add));
      }
      if (to_remove.empty()) {
        TF_CHECK_OK(ReadStringFromEnvVar(remove_env_var, "", &to_remove));
      }
    }
    if (list_name == "CLEARLIST") {
      to_add = LIST_IS_NOT_NULL(clearlist_add);
      to_remove = LIST_IS_NOT_NULL(clearlist_remove);
      if (to_add.empty()) {
        TF_CHECK_OK(ReadStringFromEnvVar(add_env_var, "", &to_add));
      }
      if (to_remove.empty()) {
        TF_CHECK_OK(ReadStringFromEnvVar(remove_env_var, "", &to_remove));
      }
    }
    if (list_name == "DENYLIST") {
      to_add = LIST_IS_NOT_NULL(denylist_add);
      to_remove = LIST_IS_NOT_NULL(denylist_remove);
      if (to_add.empty()) {
        TF_CHECK_OK(ReadStringFromEnvVar(add_env_var, "", &to_add));
      }
      if (to_remove.empty()) {
        TF_CHECK_OK(ReadStringFromEnvVar(remove_env_var, "", &to_remove));
      }
    }

#undef LIST_IS_NOT_NULL

    for (const auto& x : str_util::Split(to_add, ",")) {
      list->insert(x);
    }
    for (const auto& x : str_util::Split(to_remove, ",")) {
      list->erase(x);
    }
  }

  // Subclasses should include these on the ClearList.
  static void AddTensorListOps(gtl::FlatSet<string>* list) {
    // Note: if a data structure op (such as TensorListPopBack) is added here,
    // IsTensorListReaderOp or IsTensorListWriterOp may need to be modified
    // LINT.IfChange
    constexpr const char* tensor_list_ops[] = {
        "TensorListConcat",
        "TensorListConcatLists",
        "TensorListConcatV2",
        "TensorListFromTensor",
        "TensorListGather",
        "TensorListGetItem",
        "TensorListPopBack",
        "TensorListPushBack",
        "TensorListPushBackBatch",
        "TensorListScatter",
        "TensorListScatterIntoExistingList",
        "TensorListScatterV2",
        "TensorListSetItem",
        "TensorListSplit",
        "TensorListStack"};
    // LINT.ThenChange(//tensorflow/core/grappler/optimizers/auto_mixed_precision.cc)
    for (auto op : tensor_list_ops) {
      list->insert(op);
    }
  }
  // TODO(plugin): Add the training ops to related list.
  // such as _FusedApplyAdam or _FusedApplyMomentum.
  // The default Allow list of BF16.
  gtl::FlatSet<string> allow_list_ops = gtl::FlatSet<string>{
      "Conv2D",        "DepthwiseConv2dNative",
      "MatMul",        "BatchMatMul",
      "BatchMatMulV2", "Tanh",
      "Fill",          "OneHot",
  };

  // The default Infer list of BF16.
  gtl::FlatSet<string> infer_list_ops = gtl::FlatSet<string>{
      "Add",
      "AddN",
      "AddV2",
      "AvgPool",
      "AvgPool3D",
      "AvgPool3DGrad",
      "AvgPoolGrad",
      "BiasAdd",
      "BiasAddGrad",
      "BiasAddV1",
      "Erf",
      "Erfc",
      "FusedBatchNormV2",
      "FusedBatchNormGradV2",
      "FusedBatchNormV3",
      "FusedBatchNormGradV3",
      "LeakyRelu",
      "LeakyReluGrad",
      "Mean",
      "Mul",
      "Sub",
      "Elu",
      "EluGrad",
      "FloorDiv",
      "_FusedBatchNormEx",
      "Log",
      "Log1p",
      "LogSoftmax",
      "Prod",
      "RealDiv",
      "Reciprocal",
      "Rsqrt",
      "Selu",
      "SeluGrad",
      "Sigmoid",
      "SigmoidGrad",
      "Softmax",
      "Softplus",
      "SoftplusGrad",
      "Softsign",
      "SoftsignGrad",
      "Sqrt",
      "Square",
      "SquaredDifference",
      "Sum"
      "TanhGrad",
      "Shape",
      "ExpandDims",
  };

  // The default Deny list of BF16.
  gtl::FlatSet<string> deny_list_ops = gtl::FlatSet<string>{
      "Exp",
      "Expm1",
      "L2Loss",
      "Pow",
      "SaveV2",
      "SoftmaxCrossEntropyWithLogits",
      "SparseSoftmaxCrossEntropyWithLogits",
  };

  // The default Clear list of BF16.
  gtl::FlatSet<string> clear_list_ops = gtl::FlatSet<string>{
      "Abs",
      "ArgMax",
      "ArgMin",
      "BatchToSpace",
      "BatchToSpaceND",
      "BroadcastTo",
      "Ceil",
      "CheckNumerics",
      "ClipByValue",
      "Concat",
      "ConcatV2",
      "DepthToSpace",
      "DynamicPartition",
      "DynamicStitch",
      "EnsureShape",
      "Enter",
      "Equal",
      "Exit",
      "Floor",
      "Gather",
      "GatherNd",
      "GatherV2",
      "Greater",
      "GreaterEqual",
      "Identity",
      "IsFinite",
      "IsInf",
      "IsNan",
      "Less",
      "LessEqual",
      "Max",
      "Maximum",
      "MaxPool",
      "MaxPool3D",
      "MaxPool3DGrad",
      "MaxPoolGrad",
      "MaxPoolGradGrad",
      "MaxPoolGradGradV2",
      "MaxPoolGradV2",
      "MaxPoolV2",
      "Merge",
      "Min",
      "Minimum",
      "MirrorPad",
      "MirrorPadGrad",
      "Neg",
      "NextIteration",
      "NotEqual",
      "OnesLike",
      "Pack",
      "Pad",
      "PadV2",
      "PreventGradient",
      "Rank",
      "Relu",
      "Relu6",
      "Relu6Grad",
      "ReluGrad",
      "Reshape",
      "ResizeNearestNeighbor",
      "ResizeNearestNeighborGrad",
      "ResizeBilinear",
      "Reverse",
      "ReverseSequence",
      "ReverseV2",
      "Round",
      "ScatterNd",
      "Select",
      "SelectV2",
      "ShapeN",
      "Sign",
      "Slice",
      "Snapshot",
      "SpaceToBatch",
      "SpaceToBatchND",
      "SpaceToDepth",
      "Split",
      "SplitV",
      "Squeeze",
      "StopGradient",
      "StridedSlice",
      "StridedSliceGrad",
      "Switch",
      "Tile",
      "TopK",
      "TopKV2",
      "Transpose",
      "Where",
      "Unpack",
      "ZerosLike",
  };
};

class AutoMixedPrecisionListsCPU : public AutoMixedPrecisionLists {
 public:
  AutoMixedPrecisionListsCPU() {}

  gtl::FlatSet<string> AllowList() override {
    UpdateList("ALLOWLIST", &allow_list_ops);
    return allow_list_ops;
  }

  gtl::FlatSet<string> InferList() override {
    auto add_ops = gtl::FlatSet<string>{"Sum", "Square"};
    for (auto op : add_ops) {
      infer_list_ops.insert(op);
    }
    UpdateList("INFERLIST", &infer_list_ops);
    return infer_list_ops;
  }

  gtl::FlatSet<string> DenyList() override {
    auto remove_ops = gtl::FlatSet<string>{"Sum"};
    for (auto op : remove_ops) {
      deny_list_ops.erase(op);
    }
    UpdateList("DENYLIST", &deny_list_ops);
    return deny_list_ops;
  }

  gtl::FlatSet<string> ClearList() override {
    AddTensorListOps(&clear_list_ops);
    UpdateList("CLEARLIST", &clear_list_ops);
    return clear_list_ops;
  }
};

}  // namespace graph
}  // namespace amd_cpu_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_GRAPH_AUTO_MIXED_PRECISION_AUTO_MIXED_PRECISION_LISTS_H_
