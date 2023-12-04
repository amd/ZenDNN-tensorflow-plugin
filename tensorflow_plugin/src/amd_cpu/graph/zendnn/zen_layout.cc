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

#include "tensorflow_plugin/src/amd_cpu/graph/zendnn/zen_layout.h"

#include <regex>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow_plugin/src/amd_cpu/graph/utils/graph_properties.h"
#include "tensorflow_plugin/src/amd_cpu/graph/utils/op_types.h"
#include "tensorflow_plugin/src/amd_cpu/graph/utils/utils.h"
#include "tensorflow_plugin/src/amd_cpu/util/attr_value_util.h"
#include "tensorflow_plugin/src/amd_cpu/util/types.h"

namespace amd_cpu_plugin {
namespace graph {

namespace {

const std::vector<ZenFormatInfo>* GetZenFormatInfo() {
  static std::vector<ZenFormatInfo> rinfo{
      {"Conv2D", "_ZenConv2D", CopyAttrsZenConv2D, AlwaysRewrite},
      {"DepthwiseConv2dNative", "_ZenDepthwiseConv2dNative", CopyAttrsZenConv2D,
       AlwaysRewrite},
      {"_FusedConv2D", "_ZenFusedConv2D", CopyAttrsZenFusedConv2D,
       RewriteFusedConv2D},
      {"_FusedDepthwiseConv2dNative", "_ZenFusedDepthwiseConv2dNative",
       CopyAttrsZenFusedConv2D, RewriteFusedConv2D},
      {"MaxPool", "_ZenMaxPool", CopyAttrsAll, AlwaysRewrite},
      {"AvgPool", "_ZenAvgPool", CopyAttrsAll, AlwaysRewrite},
      {"MatMul", "_ZenMatMul", CopyAttrsAll, AlwaysRewrite},
      {"_FusedMatMul", "_ZenFusedMatMul", CopyAttrsAll, AlwaysRewrite},
      {"BatchMatMul", "_ZenBatchMatMul", CopyAttrsAll, AlwaysRewrite},
      {"BatchMatMulV2", "_ZenBatchMatMulV2", CopyAttrsAll, AlwaysRewrite},
      {"_FusedBatchMatMulV2", "_ZenFusedBatchMatMulV2", CopyAttrsAll,
       AlwaysRewrite},
      // We are not supporting BLOCKED format execution.
      {"FusedBatchNorm", "_ZenFusedBatchNorm", CopyAttrsAll, AlwaysRewrite},
      {"FusedBatchNormV2", "_ZenFusedBatchNormV2", CopyAttrsAll, AlwaysRewrite},
      {"FusedBatchNormV3", "_ZenFusedBatchNormV3", CopyAttrsAll, AlwaysRewrite},
      {"Softmax", "_ZenSoftmax", CopyAttrsAll, AlwaysRewrite},
      {"InvertPermutation", "_ZenInvertPermutation", CopyAttrsAll,
       AlwaysRewrite},
      {"Transpose", "_ZenTranspose", CopyAttrsAll, AlwaysRewrite},
      {"ConjugateTranspose", "_ZenConjugateTranspose", CopyAttrsAll,
       AlwaysRewrite}};
  return &rinfo;
}
}  // namespace

const ZenFormatInfo* CheckForNodeZenFormat(
    const utils::MutableNodeView& node_view) {
  NodeDef& node_def = *(node_view.node());
  if (!IsLayoutRewriteSupportedDataType(node_def)) return nullptr;

  // We now check if rewrite rule applies for this op. If rewrite rule passes
  // for this op, then we rewrite it to Zen op.
  // Find matching ZenFormatInfo and then check that rewrite rule applies.
  const std::vector<ZenFormatInfo>* rinfo = GetZenFormatInfo();
  for (auto ri = rinfo->cbegin(); ri != rinfo->cend(); ++ri) {
    if (node_def.op() == ri->name && ri->rewrite_rule(node_view)) {
      return &*ri;
    }
  }

  // Else return not found.
  return nullptr;
}

// Rewrites input node to a new node specified by its matching rewrite info.
//
// Method first searches matching rewrite info for input node and then
// uses that info to rewrite.
//
// Input node may be deleted in case of rewrite. Attempt to use the node
// after the call can result in undefined behaviors.
Status RewriteNode(ZenFormatContext* ctx, const int node_index,
                   const ZenFormatInfo* ri, const NodeMap& node_map) {
  const auto* node_view = ctx->graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();

  NodeDef new_node_def;
  // Let's copy all inputs (TF tensors) of original node to new node.
  for (int idx = 0; idx < node_view->NumRegularFanins(); idx++) {
    new_node_def.add_input(node_def->input(idx));
  }

  new_node_def.set_name(node_def->name());
  new_node_def.set_op(ri->new_name);
  new_node_def.set_device(node_def->device());

  ri->copy_attrs(node_view, &new_node_def);

  // Non-control inputs also include const type inputs like "kernel" and "bias".
  // For pluggable device, NodeDef can only provide name of the inputs, which
  // makes it difficult to differentiate, as in_links is not used, it does not
  // affect execution of mempool.
  int num_non_control_inputs = NumNonControlInputs(*node_def);
  // Non-control outputs are correctly configured, but the graph provided to
  // plugin, does not contain "_Retval" node at this phase, hence we increment
  // the out_links value for the leaf nodes.
  int num_non_control_outputs = NumNonControlOutputs((*node_def), node_map);
  if (num_non_control_outputs==0) num_non_control_outputs++;

  AddNodeAttr("in_links", num_non_control_inputs, &new_node_def);
  AddNodeAttr("out_links", num_non_control_outputs, &new_node_def);

  // Incoming data edges from 'orig_node' node to new 'new_node' node are
  // already copied in BuildNode. We need to handle control edges now.
  for (int idx = 0; idx < node_view->NumControllingFanins(); idx++) {
    new_node_def.add_input(
        node_def->input(node_view->NumRegularFanins() + idx));
  }

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();

  // apply mutation.
  Status status;
  mutation->AddNode(std::move(new_node_def), &status);
  TF_ABORT_IF_ERROR(std::move(status));
  TF_ABORT_IF_ERROR(mutation->Apply());
  return OkStatus();
}

Status RunZenLayout(const char* device_name, const GrapplerItem& item,
                    const GraphDef& graph_def, GraphDef* optimized_graph) {
  Status status;
  GraphDef multable_graph_def = graph_def;
  NodeMap node_map(&multable_graph_def);
  ZenFormatContext ctx(item, &multable_graph_def, &status);

  // Processing graph in reverse-topological sorted order allows to remap
  // longer chains of dependent ops in one pass.
  TF_ABORT_IF_ERROR(
      ctx.graph_view.SortTopologically(/*ignore_cycles=*/false, {}));

  // Skip nodes that were invalidated.
  int num_nodes = multable_graph_def.node_size();

  zendnnInfo(ZENDNN_FWKLOG, "ZenLayoutPass: Start to rewrite nodes.");

  for (int node_index = num_nodes - 1; node_index >= 0; --node_index) {
    const auto* node_view = ctx.graph_view.GetNode(node_index);
    const auto* node_def = node_view->node();
    // Check if node can run on current optimizer device.
    if (!NodeIsOnDevice(device_name, node_def)) continue;
    // Don't rewrite fetch node because must keep its name unchanged.
    const ZenFormatInfo* ri = nullptr;
    // We will first search if node is to be rewritten.
    if ((ri = CheckForNodeZenFormat(*node_view)) != nullptr) {
      const string& node_name = node_def->name();
      const string& op_name = node_def->op();

      if (RewriteNode(&ctx, node_index, ri, node_map) == OkStatus()) {
        zendnnInfo(ZENDNN_FWKLOG, "ZenLayoutPass: rewrote node ", node_name,
                   " with op ", op_name, " for Zen layout optimization.");
      } else {
        zendnnInfo(ZENDNN_FWKLOG, "ZenLayoutPass: found node ", node_name,
                   " with op ", op_name, " but rewrite failed.");
      }
    }
  }

  // Setting the reset value of last Zen node to true.
  for (int node_index = num_nodes - 1; node_index >= 0; --node_index) {
    const auto* node_view = ctx.graph_view.GetNode(node_index);
    const auto* node_def = node_view->node();
    const string& op_name = node_def->op();

    if (regex_search(op_name, std::regex("^_Zen"))) {
      const AttrValue* attr = node_view->GetAttr("reset");
      SetAttrValue(true, const_cast<AttrValue*>(attr));
      break;
    }
  }

  *optimized_graph = std::move(multable_graph_def);
  return OkStatus();
}

}  // namespace graph
}  // namespace amd_cpu_plugin
