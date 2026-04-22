/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
 ******************************************************************************/

// TensorFlow C API headers.
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_status.h"
// TensorFlow plug-in headers.
#include "tensorflow_plugin/src/amd_cpu/ops/zendnn/shape_inference_fns.h"
#include "tensorflow_plugin/src/amd_cpu/util/zen_utils.h"
// ZenDNNL logging support
#include "common/zendnnl_global.hpp"

namespace amd_cpu_plugin {

// Routine for registering _ZenSafeEmbeddingLookupSparse op.
//
// This fused op replaces the entire safe_embedding_lookup_sparse subgraph:
//   SparseReshape -> GatherV2 (indices/values) -> SparseFillEmptyRows ->
//   SparseSegment{Sum,Mean,SqrtN} -> SelectV2 (zero_empty_rows) -> Reshape
//
// Inputs:
//   0: params      - Embedding table [num_embeddings, embedding_dim]
//   1: sp_indices   - Sparse tensor indices [nnz, rank]
//   2: sp_values    - Sparse tensor values (indices into params) [nnz]
//   3: sp_dense_shape - Dense shape of the sparse tensor [rank]
//   4: default_value  - Default fill value for empty rows (scalar)
//
// Outputs:
//   0: output - Embedded output [batch_size, embedding_dim]
//
// Attrs:
//   combiner: "sum", "mean", or "sqrtn"
//   T: float or bfloat16 (embedding table type)
//   Tindices: int32 or int64 (sparse values / indices type)
void RegisterZenFusedEmbeddingLookupSparse() {
  TF_Status* status = TF_NewStatus();

  TF_OpDefinitionBuilder* op_builder =
      TF_NewOpDefinitionBuilder("_ZenSafeEmbeddingLookupSparse");
  TF_OpDefinitionBuilderAddInput(op_builder, "params: T");
  TF_OpDefinitionBuilderAddInput(op_builder, "sp_indices: int64");
  TF_OpDefinitionBuilderAddInput(op_builder, "sp_values: Tindices");
  TF_OpDefinitionBuilderAddInput(op_builder, "sp_dense_shape: int64");
  TF_OpDefinitionBuilderAddInput(op_builder, "default_value: Tindices");
  TF_OpDefinitionBuilderAddInput(op_builder, "orig_dense_shape: int64");
  TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
  TF_OpDefinitionBuilderAddAttr(op_builder, "T: {float, bfloat16} = DT_FLOAT");
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                "Tindices: {int32, int64} = DT_INT64");
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                "combiner: {'sum', 'mean', 'sqrtn'} = 'mean'");
  TF_OpDefinitionBuilderAddAttr(op_builder, "has_adjust_shape: bool = false");
  TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                  &unknown_shape_fn);

  TF_RegisterOpDefinition(op_builder, status);
  if (TF_OK != TF_GetCode(status)) {
    zendnnl::error_handling::apilog_error(
        "Failed to register _ZenSafeEmbeddingLookupSparse: ",
        TF_Message(status));
  } else {
    zendnnl::error_handling::apilog_info(
        "Registered op: _ZenSafeEmbeddingLookupSparse");
  }
  TF_DeleteStatus(status);
}

}  // namespace amd_cpu_plugin

void RegisterZenEmbeddingLookupSparseOps() {
  amd_cpu_plugin::RegisterZenFusedEmbeddingLookupSparse();
}
