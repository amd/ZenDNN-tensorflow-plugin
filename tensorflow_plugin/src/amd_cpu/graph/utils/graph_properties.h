/*******************************************************************************
 * Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights
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

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_GRAPH_UTILS_GRAPH_PROPERTIES_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_GRAPH_UTILS_GRAPH_PROPERTIES_H_

#include <string>
#include <vector>

#include "protos/op_performance_data.pb.h"
#include "tensorflow_plugin/src/amd_cpu/graph/utils/grappler_item.h"
#include "tensorflow_plugin/src/amd_cpu/util/status.h"

namespace amd_cpu_plugin {
namespace graph {

class GraphProperties {
 public:
  explicit GraphProperties(const GrapplerItem& item);
  ~GraphProperties();

  // Infer the shapes through abstract interpretation. Feed information can be
  // incorrect so it should be discarded to ensure correctness of the analysis.
  // However, it can help infer shapes in the fanout of fed nodes (even though
  // the correctness of these shapes can't be guaranteed), so in some cases
  // (such as simulation or scheduling) it makes sense of keep these shapes.
  // aggressive_shape_inference option executes nodes on the host to identify
  // output values when possible and does other aggressive strategies.
  // Similar to assuming_valid_feeds, this may cause incorrectness in graph
  // analyses, but is useful for simulation or scheduling.
  // If include_input_tensor_values is true, the values of constant tensors
  // will included in the input properties.
  // If include_output_tensor_values is true, the values of constant tensors
  // will be included in the output properties.
  Status InferStatically(bool assume_valid_feeds,
                         bool aggressive_shape_inference,
                         bool include_input_tensor_values,
                         bool include_output_tensor_values);
  Status InferStatically(bool assume_valid_feeds,
                         bool aggressive_shape_inference,
                         bool include_tensor_values) {
    return InferStatically(
        assume_valid_feeds,
        /*aggressive_shape_inference=*/aggressive_shape_inference,
        /*include_input_tensor_values=*/include_tensor_values,
        /*include_output_tensor_values=*/include_tensor_values);
  }
  Status InferStatically(bool assume_valid_feeds) {
    return InferStatically(assume_valid_feeds,
                           /*aggressive_shape_inference=*/false,
                           /*include_tensor_values=*/true);
  }

  Status GetInputProperties(
      const string& node_name,
      std::vector<OpInfo_TensorProperties>* input_props) const;

  Status GetOutputProperties(
      const string& node_name,
      std::vector<OpInfo_TensorProperties>* output_props) const;

 private:
  TF_GraphProperties* graph_prop_;
};

}  // namespace graph
}  // namespace amd_cpu_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_GRAPH_UTILS_GRAPH_PROPERTIES_H_
