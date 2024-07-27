/*******************************************************************************
 * Modifications Copyright (c) 2024 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 ******************************************************************************/

/* Copyright (c) 2021-2022 Intel Corporation

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

#include "tensorflow_plugin/src/amd_cpu/graph/utils/layout_utils.h"

#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow_plugin/src/amd_cpu/graph/utils/op_types.h"
#include "tensorflow_plugin/src/amd_cpu/graph/utils/utils.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_def_util.h"

namespace amd_cpu_plugin {
namespace graph {

//////////////////////////////////////////////////////////////////////////
// Rewrite functions for Quantized ops.
//////////////////////////////////////////////////////////////////////////
void CopyAttrsQuantizedConv2D(const utils::MutableNodeView* orig_node_view,
                              NodeDef* new_node) {
  CopyAttrsAll(orig_node_view, new_node);

  // Get all attributes from old node.
  const NodeDef* orig_node_def = orig_node_view->node();
  DataType out_type;
  TF_CHECK_OK(GetNodeAttr(*orig_node_def, "out_type", &out_type));

  // Add attributes to new node.
  auto* new_attr = new_node->mutable_attr();

  // TODO(plugin): avoid hardcode "NHWC" for QuantizedConv2D.
  string data_format("NHWC");
  SetAttrValue(data_format, &(*new_attr)["data_format"]);

  // Tbias is only valid for quantized op meeting 2 requirements:
  // 1. fused with BiasAdd.
  // 2. fused with Requantize or Dequantize.
  DataType Tbias;
  if (TryGetNodeAttr(*orig_node_def, "Tbias", &Tbias)) {
    SetAttrValue(Tbias, &(*new_attr)["Tbias"]);
  }
}

void CopyAttrsQCBR(const utils::MutableNodeView* orig_node_view,
                   NodeDef* new_node) {
  DataType Tinput, Tfilter, out_type, Tbias, Tsummand;
  bool narrow_range, reset, reorder_after, reorder_before;
  string padding;
  string data_format("NHWC");

  std::vector<int32> strides, dilations, padding_list;
  const NodeDef* orig_node_def = orig_node_view->node();
  bool has_padding_list = HasNodeAttr(*orig_node_def, "padding_list");

  // Get all attributes from old node.
  TF_CHECK_OK(GetNodeAttr(*orig_node_def, "Tinput", &Tinput));
  TF_CHECK_OK(GetNodeAttr(*orig_node_def, "Tfilter", &Tfilter));
  TF_CHECK_OK(GetNodeAttr(*orig_node_def, "Tbias", &Tbias));
  TF_CHECK_OK(GetNodeAttr(*orig_node_def, "strides", &strides));
  TF_CHECK_OK(GetNodeAttr(*orig_node_def, "padding", &padding));
  TF_CHECK_OK(GetNodeAttr(*orig_node_def, "out_type", &out_type));
  TF_CHECK_OK(GetNodeAttr(*orig_node_def, "dilations", &dilations));
  if (has_padding_list) {
    TF_CHECK_OK(GetNodeAttr(*orig_node_def, "padding_list", &padding_list));
  }

  auto* new_attr = new_node->mutable_attr();
  NodeDef* filter_node = nullptr;

  // Add attributes to new node.
  SetAttrValue(Tinput, &(*new_attr)["Tinput"]);
  SetAttrValue(Tfilter, &(*new_attr)["Tfilter"]);
  SetAttrValue(out_type, &(*new_attr)["out_type"]);
  SetAttrValue(padding, &(*new_attr)["padding"]);
  SetAttrValue(strides, &(*new_attr)["strides"]);
  SetAttrValue(Tbias, &(*new_attr)["Tbias"]);
  SetAttrValue(dilations, &(*new_attr)["dilations"]);
  SetAttrValue(data_format, &(*new_attr)["data_format"]);
  SetAttrValue(reorder_before, &(*new_attr)["reorder_before"]);
  SetAttrValue(reorder_after, &(*new_attr)["reorder_after"]);
  SetAttrValue(reset, &(*new_attr)["reset"]);
  if (has_padding_list) {
    SetAttrValue(padding_list, &(*new_attr)["padding_list"]);
  }

  if (HasNodeAttr(*orig_node_def, "Tsummand")) {
    TF_CHECK_OK(GetNodeAttr(*orig_node_def, "Tsummand", &Tsummand));
    SetAttrValue(Tsummand, &(*new_attr)["Tsummand"]);
  }
}

void UpdateZenOpAttrs(const utils::MutableNodeView* orig_node_view,
                      NodeDef* new_node) {
  string name;
  DataType T, out_type, quantizedtype;
  string mode, round_mode;
  bool narrow_range, reset, reorder_after, reorder_before;
  int axis;
  float ensure_minimum_range;
  const NodeDef* orig_node_def = orig_node_view->node();
  TF_CHECK_OK(GetNodeAttr(*orig_node_def, "T", &T));
  TF_CHECK_OK(GetNodeAttr(*orig_node_def, "mode", &mode));
  TF_CHECK_OK(GetNodeAttr(*orig_node_def, "narrow_range", &narrow_range));
  TF_CHECK_OK(GetNodeAttr(*orig_node_def, "axis", &axis));

  auto* new_attr = new_node->mutable_attr();
  SetAttrValue(T, &(*new_attr)["T"]);
  SetAttrValue(mode, &(*new_attr)["mode"]);
  SetAttrValue(round_mode, &(*new_attr)["round_mode"]);
  SetAttrValue(narrow_range, &(*new_attr)["narrow_range"]);
  SetAttrValue(axis, &(*new_attr)["axis"]);
}

//////////////////////////////////////////////////////////////////////////
// Rewrite functions.
//////////////////////////////////////////////////////////////////////////

bool AlwaysRewrite(const utils::MutableNodeView& node_view) { return true; }

bool RewriteSupportedDataType(const utils::MutableNodeView& node_view) {
  const NodeDef& node_def = *(node_view.node());
  const string& op_name = node_def.op();
  DataType T;
  AttrSlice attr_list(node_def);
  if (!TryGetNodeAttr(attr_list, "T", &T)) {
    return false;
  }

  return IsLayoutRewriteSupportedDataType(op_name, T);
}

bool RewriteQuantize(const utils::MutableNodeView& node_view) {
  const NodeDef& node_def = *(node_view.node());
  const string& op_name = node_def.op();
  DataType Tinput;
  GetNodeAttr(node_def, "Tinput", &Tinput);
  if (Tinput == DT_QUINT8 || Tinput == DT_QINT8) {
    return true;
  }
}

bool RewriteFusedConv2D(const utils::MutableNodeView& node_view) {
  if (!RewriteSupportedDataType(node_view)) return false;

  const NodeDef& node_def = *(node_view.node());
  std::vector<string> fused_ops;
  GetNodeAttr(node_def, "fused_ops", &fused_ops);

  return (fused_ops == std::vector<string>{"BiasAdd"} ||
          fused_ops == std::vector<string>{"BiasAdd", "Relu"} ||
          fused_ops == std::vector<string>{"BiasAdd", "Relu6"} ||
          fused_ops == std::vector<string>{"BiasAdd", "LeakyRelu"} ||
          fused_ops == std::vector<string>{"BiasAdd", "Add"} ||
          fused_ops == std::vector<string>{"BiasAdd", "Add", "Relu"} ||
          fused_ops == std::vector<string>{"FusedBatchNorm"} ||
          fused_ops == std::vector<string>{"FusedBatchNorm", "Relu"} ||
          fused_ops == std::vector<string>{"FusedBatchNorm", "LeakyRelu"});
}

bool RewriteFusedMatMul(const utils::MutableNodeView& node_view) {
  if (!RewriteSupportedDataType(node_view)) return false;
  const NodeDef& node_def = *(node_view.node());
  std::vector<string> fused_ops;
  GetNodeAttr(node_def, "fused_ops", &fused_ops);

  return (fused_ops == std::vector<string>{"BiasAdd"} ||
          fused_ops == std::vector<string>{"BiasAdd", "Add"} ||
          fused_ops == std::vector<string>{"BiasAdd", "Relu"} ||
          fused_ops == std::vector<string>{"BiasAdd", "Add", "Relu"} ||
          fused_ops == std::vector<string>{"BiasAdd", "GeluExact"} ||
          fused_ops == std::vector<string>{"BiasAdd", "GeluApproximate"});
}

//////////////////////////////////////////////////////////////////////////
// Op-specific functions to copy attributes from old node to new node.
//////////////////////////////////////////////////////////////////////////

void CopyAttrsAll(const utils::MutableNodeView* orig_node_view,
                  NodeDef* new_node) {
  // Setup ZenDNN specific attributes.
  CopyZenAttrs(*(orig_node_view->node()), new_node);

  CopyAllAttrs(*(orig_node_view->node()), new_node);
}

// Copies the attributes from Conv2D op to _ZenConv2D op. 'padding' and
// 'explicit_paddings' attributes are updated accordingly to PadConv2D fusion.
void CopyAttrsZenConv2D(const utils::MutableNodeView* orig_node_view,
                        NodeDef* new_node) {
  DataType T;
  string data_format;
  string padding;
  std::vector<int32> strides;
  std::vector<int32> dilations;
  std::vector<int32> explicit_paddings = {};

  // Setup ZenDNN specific attributes.
  CopyZenAttrs(*(orig_node_view->node()), new_node);

  // Get all attributes from old node.
  const NodeDef* orig_node_def = orig_node_view->node();

  // Get attributes from TF op node.
  TF_CHECK_OK(GetNodeAttr(*orig_node_def, "T", &T));
  TF_CHECK_OK(GetNodeAttr(*orig_node_def, "strides", &strides));
  TF_CHECK_OK(GetNodeAttr(*orig_node_def, "padding", &padding));
  TF_CHECK_OK(GetNodeAttr(*orig_node_def, "data_format", &data_format));
  TF_CHECK_OK(GetNodeAttr(*orig_node_def, "dilations", &dilations));

  // 'padding_update' determines if padding attributes needs to be modified.
  bool padding_update = false;
  // TODO(plugin) : Add pad fusion later.
  // PadConv2D fusion can be done for VALID and EXPLICIT padding.
  // if (padding != "SAME") {
  // Check if PadConv2D fusion can be done and get the padding values.
  // padding_update = UpdateAttributePadConv2D(padding, orig_node_def,
  // explicit_paddings);
  // }
  // Update Zen op with attributes from TF op.
  auto* new_attr = new_node->mutable_attr();
  SetAttrValue(T, &(*new_attr)["T"]);
  SetAttrValue(strides, &(*new_attr)["strides"]);
  // Update 'padding' attribute for PadConv2D fusion.
  if (padding_update == true) {
    SetAttrValue("EXPLICIT", &(*new_attr)["padding"]);  // Updates padding type.
    SetAttrValue(explicit_paddings,
                 &(*new_attr)["explicit_paddings"]);  // sets padding values.
  } else {
    // 'padding' attribute for condition when fusion is not performed.
    SetAttrValue(padding, &(*new_attr)["padding"]);
    // If 'padding' is EXPLICIT, then 'explicit_paddings' attribute needs to be
    // set.
    if (padding == "EXPLICIT") {
      std::vector<int32> explicit_paddings_tmp = {};
      TF_CHECK_OK(GetNodeAttr(*orig_node_def, "explicit_paddings",
                              &explicit_paddings_tmp));
      SetAttrValue(explicit_paddings_tmp, &(*new_attr)["explicit_paddings"]);
    }
  }
  SetAttrValue(data_format, &(*new_attr)["data_format"]);
  SetAttrValue(dilations, &(*new_attr)["dilations"]);
}

// Copy the attributes from FusedConv2D op to _ZenFusedConv2D op. 'padding' and
// 'explicit_paddings' attributes are updated according to PadFusedConv2D
// fusion.
void CopyAttrsZenFusedConv2D(const utils::MutableNodeView* orig_node_view,
                             NodeDef* new_node) {
  DataType T;
  int num_args;
  float epsilon;
  float leakyrelu_alpha;
  string data_format;
  string padding;
  std::vector<string> fused_ops;
  std::vector<int32> strides;
  std::vector<int32> dilations;
  std::vector<int32> explicit_paddings = {};

  // Setup ZenDNN specific attributes.
  CopyZenAttrs(*(orig_node_view->node()), new_node);

  // Get all attributes from old node.
  const NodeDef* orig_node_def = orig_node_view->node();

  // Get attributes from TF op node.
  TF_CHECK_OK(GetNodeAttr(*orig_node_def, "T", &T));
  TF_CHECK_OK(GetNodeAttr(*orig_node_def, "num_args", &num_args));
  TF_CHECK_OK(GetNodeAttr(*orig_node_def, "fused_ops", &fused_ops));
  TF_CHECK_OK(GetNodeAttr(*orig_node_def, "strides", &strides));
  TF_CHECK_OK(GetNodeAttr(*orig_node_def, "padding", &padding));
  TF_CHECK_OK(GetNodeAttr(*orig_node_def, "data_format", &data_format));
  TF_CHECK_OK(GetNodeAttr(*orig_node_def, "dilations", &dilations));
  TF_CHECK_OK(GetNodeAttr(*orig_node_def, "epsilon", &epsilon));
  if (HasNodeAttr(*orig_node_def, "leakyrelu_alpha")) {
    TF_CHECK_OK(
        GetNodeAttr(*orig_node_def, "leakyrelu_alpha", &leakyrelu_alpha));
  }

  // 'padding_update' determines if padding attributes needs to be modified.
  bool padding_update = false;
  // TODO(plugin) : Add pad fusion later.
  // PadFusedConv2D fusion can be done for VALID and EXPLICIT padding.
  // if (padding != "SAME") {
  // Check if PadFusedConv2D fusion can be done and get the padding values.
  // padding_update =
  //     UpdateAttributePadConv2D(padding, orig_node, explicit_paddings);
  // }
  // Update Zen op with attributes from TF op.
  auto* new_attr = new_node->mutable_attr();
  SetAttrValue(T, &(*new_attr)["T"]);
  SetAttrValue(strides, &(*new_attr)["strides"]);
  SetAttrValue(num_args, &(*new_attr)["num_args"]);
  SetAttrValue(fused_ops, &(*new_attr)["fused_ops"]);

  // Update padding attribute for PadConv2D fusion.
  if (padding_update == true) {
    SetAttrValue("EXPLICIT", &(*new_attr)["padding"]);  // Updates padding type.
    SetAttrValue(explicit_paddings,
                 &(*new_attr)["explicit_paddings"]);  // sets padding values.
  } else {
    // 'padding' attribute for condition when fusion is not performed.
    SetAttrValue(padding, &(*new_attr)["padding"]);
    // If 'padding' is EXPLICIT, then 'explicit_paddings' attribute needs to be
    // set.
    if (padding == "EXPLICIT") {
      std::vector<int32> explicit_paddings_tmp = {};
      TF_CHECK_OK(GetNodeAttr(*orig_node_def, "explicit_paddings",
                              &explicit_paddings_tmp));
      SetAttrValue(explicit_paddings_tmp, &(*new_attr)["explicit_paddings"]);
    }
  }
  SetAttrValue(data_format, &(*new_attr)["data_format"]);
  SetAttrValue(dilations, &(*new_attr)["dilations"]);
  SetAttrValue(epsilon, &(*new_attr)["epsilon"]);
  if (HasNodeAttr(*orig_node_def, "leakyrelu_alpha")) {
    SetAttrValue(leakyrelu_alpha, &(*new_attr)["leakyrelu_alpha"]);
  }
}

bool IsLayoutRewriteSupportedDataType(const string& op_name,
                                      const DataType& T) {
  if (op_name == "Reshape" || op_name == "Transpose") {
    return (T == DT_INT32) || (T == DT_INT64) || (T == DT_COMPLEX64) ||
           (T == DT_COMPLEX128) || (T == DT_FLOAT);
  } else if (op_name == "InvertPermutation") {
    return (T == DT_INT32) || (T == DT_INT64);
  } else if (op_name == "ConjugateTranspose") {
    return (T == DT_COMPLEX64) || (T == DT_COMPLEX128);
  } else if (op_name == "Mul" || op_name == "Sub" ||
             op_name == "SquaredDifference" || op_name == "Add" ||
             op_name == "AddV2" || op_name == "Maximum") {
    return (T == DT_FLOAT);
  } else if (op_name == "FusedBatchNorm" || op_name == "FusedBatchNormV2" ||
             op_name == "FusedBatchNormV3" || op_name == "_FusedBatchNormEx") {
    return (T == DT_FLOAT);
  } else {
    return (T == DataType::DT_FLOAT || T == DataType::DT_BFLOAT16);
  }
}

OpDef GetOpDef(const NodeDef& node_def) {
  static FunctionLibraryDefinition function_lib =
      FunctionLibraryDefinition(GraphDef());
  OpDef op_def;
  Status status = function_lib.LookUpOpDef(node_def.op(), &op_def);

  TF_ABORT_IF_ERROR(status);

  return op_def;
}

void CopyZenAttrs(const NodeDef& orig_node, NodeDef* new_node) {
  AddNodeAttr("reorder_before", false, new_node);
  AddNodeAttr("reorder_after", false, new_node);
  AddNodeAttr("reset", false, new_node);
  AddNodeAttr("is_eager", false, new_node);
}

void CopyAllAttrs(const NodeDef& orig_node, NodeDef* new_node) {
  string name;
  AttrSlice attr_list(orig_node);

  auto iter = attr_list.begin();
  OpDef op_def = GetOpDef(*new_node);

  while (iter != attr_list.end()) {
    name = iter->first;
    auto attr = iter->second;

    // Check OpDef first to exclude undefined attr in `new_node`.
    if (FindAttrMutable(name, &op_def) != nullptr) {
      AddNodeAttr(name, attr, new_node);
    }
    ++iter;
  }
}

}  // namespace graph
}  // namespace amd_cpu_plugin
