/*******************************************************************************
 * Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights
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
// Rewrite functions
//////////////////////////////////////////////////////////////////////////

bool AlwaysRewrite(const utils::MutableNodeView& node_view) { return true; }

bool RewriteFusedConv2D(const utils::MutableNodeView& node_view) {
  const NodeDef& node_def = *(node_view.node());
  std::vector<string> fused_ops;
  GetNodeAttr(node_def, "fused_ops", &fused_ops);

  return (fused_ops == std::vector<string>{"BiasAdd"} ||
          fused_ops == std::vector<string>{"FusedBatchNorm"} ||
          fused_ops == std::vector<string>{"Relu"} ||
          fused_ops == std::vector<string>{"BiasAdd", "Relu"} ||
          fused_ops == std::vector<string>{"BiasAdd", "Relu6"} ||
          fused_ops == std::vector<string>{"BiasAdd", "Add"} ||
          fused_ops == std::vector<string>{"BiasAdd", "Add", "Relu"} ||
          fused_ops == std::vector<string>{"FusedBatchNorm", "Relu"});
}

//////////////////////////////////////////////////////////////////////////
// Op-specific functions to copy attributes from old node to new node
//////////////////////////////////////////////////////////////////////////

// _OneDnnCast has another attr T.
void CopyAttrsCast(const utils::MutableNodeView* orig_node_view,
                   NodeDef* new_node) {
  CopyAttrsAll(orig_node_view, new_node);

  DataType DstT;
  TF_CHECK_OK(GetNodeAttr(*(orig_node_view->node()), "DstT", &DstT));

  // Layout pass always check datatype by attr name T, So we need add T
  // attribution for _OneDnnCast.
  auto* new_attr = new_node->mutable_attr();
  SetAttrValue(DstT, &(*new_attr)["T"]);
}

void CopyAttrsAll(const utils::MutableNodeView* orig_node_view,
                  NodeDef* new_node) {
  // Setup ZenDNN specific attributes.
  CopyZenAttrs(*(orig_node_view->node()), new_node);

  CopyAllAttrs(*(orig_node_view->node()), new_node);
}

// Copies the attributes from Conv2D op to ZenConv2D op. 'padding' and
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
  // TODO(aakar) : Add pad fusion later.
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

// Copy the attributes from FusedConv2D op to ZenFusedConv2D op. 'padding' and
// 'explicit_paddings' attributes are updated according to PadFusedConv2D
// fusion.
void CopyAttrsZenFusedConv2D(const utils::MutableNodeView* orig_node_view,
                             NodeDef* new_node) {
  DataType T;
  int num_args;
  float epsilon;
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

  // 'padding_update' determines if padding attributes needs to be modified.
  bool padding_update = false;
  // TODO(aakar) : Add pad fusion later.
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
}

bool IsLayoutRewriteSupportedDataType(const NodeDef& node_def) {
  // Prevent rewritting if current op doesn't have attr `T`. Should bypass op
  // without `T` if want to rewrite it.
  DataType T;
  AttrSlice attr_list(node_def);
  if (!TryGetNodeAttr(attr_list, "T", &T)) {
    return false;
  }

  return (T == DataType::DT_FLOAT);
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
