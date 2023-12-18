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

#include "tensorflow_plugin/src/amd_cpu/graph/cpu_optimizer.h"

#include "tensorflow/c/experimental/grappler/grappler.h"
#include "tensorflow_plugin/src/amd_cpu/graph/auto_mixed_precision/auto_mixed_precision.h"
#include "tensorflow_plugin/src/amd_cpu/graph/remapper/remapper.h"
#include "tensorflow_plugin/src/amd_cpu/graph/utils/utils.h"
#include "tensorflow_plugin/src/amd_cpu/graph/zendnn/zen_layout.h"
#include "tensorflow_plugin/src/amd_cpu/util/errors.h"
#include "tensorflow_plugin/src/amd_cpu/util/tf_buffer.h"
#include "tensorflow_plugin/src/amd_cpu/util/util.h"

namespace amd_cpu_plugin {
namespace graph {

void* Optimizer_Create() {
  auto* optimizer = new Optimizer;
  optimizer->device_name = DEVICE_CPU;
  return reinterpret_cast<void*>(optimizer);
}

void Optimizer_Destroy(void* optimizer) {
  if (optimizer) delete reinterpret_cast<Optimizer*>(optimizer);
}

void Optimizer_Optimize(void* optimizer, const TF_Buffer* graph_buf,
                        const TF_GrapplerItem* tf_item,
                        TF_Buffer* optimized_graph_buf, TF_Status* tf_status) {
  Status status;
  // Get GrapplerItem.
  GrapplerItem item(tf_item);
  // Deserialize graph_buf into GraphDef.
  GraphDef graph_def;
  SET_STATUS_IF_ERROR(tf_status, BufferToMessage(graph_buf, graph_def));
  GraphDef optimized_graph_def = graph_def;
  if (IsZenDnnEnabled()) {
    if (IsZenDnnBF16Enabled()) {
      SET_STATUS_IF_ERROR(tf_status,
                          RunAutoMixedPrecision(
                              (static_cast<Optimizer*>(optimizer))->device_name,
                              item, graph_def, &optimized_graph_def));
      optimized_graph_def.Swap(&graph_def);
    }
    // Optimizations like Gelu require fused ops such as _FusedMatMul,
    // thus we need two sets of passes of remapper, one to generate fused ops
    // and the next to replace set patterns.
    for (int i = 0; i < 2; i++) {
      // Executing fusions before layout pass opts, i.e before Zen layout pass
      // if that is enabled.
      SET_STATUS_IF_ERROR(
          tf_status,
          RunRemapper((static_cast<Optimizer*>(optimizer))->device_name, item,
                      graph_def, &optimized_graph_def));
      optimized_graph_def.Swap(&graph_def);
    }
    DumpGraphDefToFile("remapped_graph", graph_def, "./");
    SET_STATUS_IF_ERROR(
        tf_status,
        RunZenLayout((static_cast<Optimizer*>(optimizer))->device_name, item,
                     graph_def, &optimized_graph_def));
  }
  // Serialize output GraphDef into optimized_graph_buf.
  SET_STATUS_IF_ERROR(
      tf_status, MessageToBuffer(optimized_graph_def, optimized_graph_buf));
  TF_StatusFromStatus(status, tf_status);
}

}  // namespace graph
}  // namespace amd_cpu_plugin
