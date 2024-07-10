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

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_GRAPH_UTILS_LAYOUT_UTILS_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_GRAPH_UTILS_LAYOUT_UTILS_H_

#include <string>
#include <unordered_set>

#include "tensorflow_plugin/src/amd_cpu/graph/utils/function.h"
#include "tensorflow_plugin/src/amd_cpu/graph/utils/graph_view.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_def_util.h"

namespace amd_cpu_plugin {
namespace graph {

//////////////////////////////////////////////////////////////////////////
// Rewrite functions
//////////////////////////////////////////////////////////////////////////

// Returns true if the rewite is supported for the data type.
bool RewriteSupportedDataType(const utils::MutableNodeView& node_view);

// FusedConv2D is rewritten only for a limited number of fused post-ops.
bool RewriteFusedConv2D(const utils::MutableNodeView& node_view);

// FusedMatMul is rewritten only for a limited number of fused post-ops.
bool RewriteFusedMatMul(const utils::MutableNodeView& node_view);

//////////////////////////////////////////////////////////////////////////
// Op-specific functions to copy attributes from old node to new node
//////////////////////////////////////////////////////////////////////////

// Generic function to copy all attributes from original node to target.
// graph_view is needed to get information from input node of orig_node
void CopyAttrsAll(const utils::MutableNodeView* orig_node_view,
                  NodeDef* new_node);

void CopyAttrsZenConv2D(const utils::MutableNodeView* orig_node_view,
                        NodeDef* new_node);

void CopyAttrsZenFusedConv2D(const utils::MutableNodeView* orig_node_view,
                             NodeDef* new_node);

//////////////////////////////////////////////////////////////////////////
// Helper function to handle layout process
//////////////////////////////////////////////////////////////////////////

// Copy all zen specific attributes.
void CopyZenAttrs(const NodeDef& orig_node, NodeDef* new_node);

// Returns true if rewite of "op_name" is supported with data type "T".
bool IsLayoutRewriteSupportedDataType(const string& op_name, const DataType& T);

// Sub function to copy attrs from original node to new node.
void CopyAllAttrs(const NodeDef& orig_node, NodeDef* new_node);

OpDef GetOpDef(const NodeDef& node_def);

}  // namespace graph
}  // namespace amd_cpu_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_GRAPH_UTILS_LAYOUT_UTILS_H_
