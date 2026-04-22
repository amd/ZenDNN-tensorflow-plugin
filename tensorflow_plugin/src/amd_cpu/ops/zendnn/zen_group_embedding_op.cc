/*******************************************************************************
 * Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *******************************************************************************/

// TensorFlow C API headers.
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_status.h"
// TensorFlow plug-in headers.
#include "tensorflow_plugin/src/amd_cpu/ops/zendnn/shape_inference_fns.h"
#include "tensorflow_plugin/src/amd_cpu/util/zen_utils.h"
// ZenDNNL logging support
#include "common/zendnnl_global.hpp"

namespace amd_cpu_plugin {

void RegisterZenGroupEmbedding() {
  TF_Status* status = TF_NewStatus();

  TF_OpDefinitionBuilder* op_builder =
      TF_NewOpDefinitionBuilder("_ZenGroupEmbedding");
  TF_OpDefinitionBuilderAddInput(op_builder, "tables: num_tables * T_table");
  TF_OpDefinitionBuilderAddInput(op_builder, "indices: N * T_indices");
  TF_OpDefinitionBuilderAddInput(op_builder,
                                 "passthrough: num_passthrough * T_output");
  TF_OpDefinitionBuilderAddOutput(op_builder, "output: T_output");
  TF_OpDefinitionBuilderAddAttr(op_builder, "num_tables: int >= 1");
  TF_OpDefinitionBuilderAddAttr(op_builder, "N: int >= 1");
  TF_OpDefinitionBuilderAddAttr(op_builder, "num_passthrough: int >= 0 = 0");
  TF_OpDefinitionBuilderAddAttr(op_builder, "T_table: {float, bfloat16}");
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                "T_indices: {int32, int64} = DT_INT64");
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                "T_output: {float, bfloat16} = DT_BFLOAT16");
  TF_OpDefinitionBuilderAddAttr(op_builder, "embedding_dim: int = -1");
  TF_OpDefinitionBuilderAddAttr(op_builder, "gather_axis: int = 0");
  TF_OpDefinitionBuilderAddAttr(op_builder, "gathers_per_table: list(int)");
  // TODO (plugin): Implement proper shape inference
  // Output shape is deterministic: [outer_dims..., total_features]
  // where total_features = sum(gathers_per_table[i] * num_indices *
  // embedding_dim[i]) + passthrough
  TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                  &unknown_shape_fn);

  TF_RegisterOpDefinition(op_builder, status);
  if (TF_OK != TF_GetCode(status)) {
    zendnnl::error_handling::apilog_error(
        "Failed to register _ZenGroupEmbedding: ", TF_Message(status));
  } else {
    zendnnl::error_handling::apilog_info("Registered op: _ZenGroupEmbedding");
  }
  TF_DeleteStatus(status);
}

}  // namespace amd_cpu_plugin

void RegisterZenGroupEmbeddingOps() {
  amd_cpu_plugin::RegisterZenGroupEmbedding();
}
