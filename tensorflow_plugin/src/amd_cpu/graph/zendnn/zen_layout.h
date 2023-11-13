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

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_GRAPH_ZENDNN_ZEN_LAYOUT_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_GRAPH_ZENDNN_ZEN_LAYOUT_H_

#include <string>
#include <unordered_set>
#include <vector>

#include "protos/graph.pb.h"
#include "tensorflow_plugin/src/amd_cpu/graph/utils/graph_view.h"
#include "tensorflow_plugin/src/amd_cpu/graph/utils/grappler_item.h"
#include "tensorflow_plugin/src/amd_cpu/graph/utils/layout_utils.h"
#include "tensorflow_plugin/src/amd_cpu/graph/utils/node_type_attr_map.h"
#include "tensorflow_plugin/src/amd_cpu/graph/utils/utils.h"
#include "tensorflow_plugin/src/amd_cpu/util/node_def_util.h"
#include "tensorflow_plugin/src/amd_cpu/util/zen_utils.h"

namespace amd_cpu_plugin {
namespace graph {

struct ZenFormatContext {
  explicit ZenFormatContext(const GrapplerItem& item, GraphDef* g_def,
                            Status* status)
      : graph_view(g_def, status), nodes_to_preserve(item.NodesToPreserve()) {
    TF_ABORT_IF_ERROR(node_type_map.Init(*g_def));
  }

  utils::MutableGraphView graph_view;
  std::unordered_set<string> nodes_to_preserve;
  NodeTypeAttrMap node_type_map;
};

/// Structure to specify the name of an original node, its new name after
/// rewrite, the number of inputs to the original node, the function to
/// be used to copy attributes for the op, and the rule (if any) which
/// must hold for rewriting the node.
typedef struct {
  string name;      // Original name of op of the node in the graph.
  string new_name;  // New name of the op of the node in the graph.
  // A function handler to copy attributes from an old node to a new node.
  std::function<void(const utils::MutableNodeView*, NodeDef*)> copy_attrs;
  // A rule under which to rewrite this node.
  std::function<bool(const utils::MutableNodeView&)> rewrite_rule;
} ZenFormatInfo;

const ZenFormatInfo* CheckForNodeZenFormat(
    const utils::MutableNodeView& node_view);

Status RewriteNode(ZenFormatContext* ctx, int node_index,
                   const ZenFormatInfo* ri, const NodeMap& node_map);

Status RunZenLayout(const char* device_name, const GrapplerItem& item,
                    const GraphDef& graph_def, GraphDef* optimized_graph);

}  // namespace graph
}  // namespace amd_cpu_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_GRAPH_ZENDNN_ZEN_LAYOUT_H_
