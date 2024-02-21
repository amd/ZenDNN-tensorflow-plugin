/*******************************************************************************
 * Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "tensorflow_plugin/src/amd_cpu/util/padding.h"
#include "tensorflow_plugin/src/amd_cpu/util/tensor_format.h"
#include "tensorflow_plugin/src/amd_cpu/util/zen_utils.h"

namespace amd_cpu_plugin {

// Routine for registering _ZenMatMul op.
void RegisterZenMatMul() {
  TF_Status* status = TF_NewStatus();

  TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder("_ZenMatMul");
  TF_OpDefinitionBuilderAddInput(op_builder, "a: T");
  TF_OpDefinitionBuilderAddInput(op_builder, "b: T");
  TF_OpDefinitionBuilderAddOutput(op_builder, "product: T");
  TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_a: bool = false");
  TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_b: bool = false");
  TF_OpDefinitionBuilderAddAttr(op_builder, "T: {float, bfloat16} = DT_FLOAT");
  TF_OpDefinitionBuilderAddAttr(op_builder, "is_eager: bool = false");
  TF_OpDefinitionBuilderAddAttr(op_builder, "reorder_before: bool");
  TF_OpDefinitionBuilderAddAttr(op_builder, "reorder_after: bool");
  TF_OpDefinitionBuilderAddAttr(op_builder, "in_links: int");
  TF_OpDefinitionBuilderAddAttr(op_builder, "out_links: int");
  TF_OpDefinitionBuilderAddAttr(op_builder, "reset: bool");
  TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                  &unknown_shape_fn);

  TF_RegisterOpDefinition(op_builder, status);
  if (TF_OK != TF_GetCode(status)) {
    zendnnInfo(ZENDNN_FWKLOG, "ZEN-OP-REG: _ZenMatMul Op Registration Failed!");
  } else {
    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-REG: _ZenMatMul Op Registration Is Successful!");
  }
  TF_DeleteStatus(status);
}

// Routine for registering _ZenFusedMatMul op.
void RegisterZenFusedMatMul() {
  TF_Status* status = TF_NewStatus();

  TF_OpDefinitionBuilder* op_builder =
      TF_NewOpDefinitionBuilder("_ZenFusedMatMul");
  TF_OpDefinitionBuilderAddInput(op_builder, "a: T");
  TF_OpDefinitionBuilderAddInput(op_builder, "b: T");
  TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * T");
  TF_OpDefinitionBuilderAddOutput(op_builder, "product: T");
  TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_a: bool = false");
  TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_b: bool = false");
  TF_OpDefinitionBuilderAddAttr(op_builder, "is_reshape: bool = false");
  TF_OpDefinitionBuilderAddAttr(op_builder, "T: {float, bfloat16} = DT_FLOAT");
  TF_OpDefinitionBuilderAddAttr(op_builder, "num_args: int >= 0");
  TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
  TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
  TF_OpDefinitionBuilderAddAttr(op_builder, "leakyrelu_alpha: float = 0.2");
  TF_OpDefinitionBuilderAddAttr(op_builder, "is_eager: bool = false");
  TF_OpDefinitionBuilderAddAttr(op_builder, "reorder_before: bool");
  TF_OpDefinitionBuilderAddAttr(op_builder, "reorder_after: bool");
  TF_OpDefinitionBuilderAddAttr(op_builder, "in_links: int");
  TF_OpDefinitionBuilderAddAttr(op_builder, "out_links: int");
  TF_OpDefinitionBuilderAddAttr(op_builder, "reset: bool");
  TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                  &unknown_shape_fn);

  TF_RegisterOpDefinition(op_builder, status);
  if (TF_OK != TF_GetCode(status)) {
    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-REG: _ZenFusedMatMul Op Registration Failed!");
  } else {
    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-REG: _ZenFusedMatMul Op Registration Is Successful!");
  }
  TF_DeleteStatus(status);
}

// Routine for registering _ZenMatMulBiasAddGelu op.
void RegisterZenMatMulBiasAddGelu() {
  TF_Status* status = TF_NewStatus();

  TF_OpDefinitionBuilder* op_builder =
      TF_NewOpDefinitionBuilder("_ZenMatMulBiasAddGelu");
  TF_OpDefinitionBuilderAddInput(op_builder, "a: T");
  TF_OpDefinitionBuilderAddInput(op_builder, "b: T");
  TF_OpDefinitionBuilderAddInput(op_builder, "bias: T");
  TF_OpDefinitionBuilderAddOutput(op_builder, "product: T");
  TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_a: bool = false");
  TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_b: bool = false");
  TF_OpDefinitionBuilderAddAttr(op_builder, "T: {float, bfloat16}");
  TF_OpDefinitionBuilderAddAttr(op_builder, "is_eager: bool = false");
  TF_OpDefinitionBuilderAddAttr(op_builder, "reorder_before: bool");
  TF_OpDefinitionBuilderAddAttr(op_builder, "reorder_after: bool");
  TF_OpDefinitionBuilderAddAttr(op_builder, "in_links: int");
  TF_OpDefinitionBuilderAddAttr(op_builder, "out_links: int");
  TF_OpDefinitionBuilderAddAttr(op_builder, "reset: bool");
  // TODO(plugin) :: Update shape inference function with
  // shape_inference::MatMulShape.
  TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                  &unknown_shape_fn);

  TF_RegisterOpDefinition(op_builder, status);
  if (TF_OK != TF_GetCode(status)) {
    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-REG: _ZenMatMulBiasAddGelu Op Registration Failed!");
  } else {
    zendnnInfo(
        ZENDNN_FWKLOG,
        "ZEN-OP-REG: _ZenMatMulBiasAddGelu Op Registration Is Successful!");
  }
  TF_DeleteStatus(status);
}

// Routine for registering MatMulBiasAddGelu op.
void RegisterMatMulBiasAddGelu() {
  TF_Status* status = TF_NewStatus();

  TF_OpDefinitionBuilder* op_builder =
      TF_NewOpDefinitionBuilder("MatMulBiasAddGelu");
  TF_OpDefinitionBuilderAddInput(op_builder, "a: T");
  TF_OpDefinitionBuilderAddInput(op_builder, "b: T");
  TF_OpDefinitionBuilderAddInput(op_builder, "bias: T");
  TF_OpDefinitionBuilderAddOutput(op_builder, "product: T");
  TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_a: bool = false");
  TF_OpDefinitionBuilderAddAttr(op_builder, "transpose_b: bool = false");
  TF_OpDefinitionBuilderAddAttr(op_builder, "T: {float, bfloat16}");
  // TODO(plugin) :: Update shape inference function with
  // shape_inference::MatMulShape.
  TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                  &unknown_shape_fn);

  TF_RegisterOpDefinition(op_builder, status);
  if (TF_OK != TF_GetCode(status)) {
    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-REG: MatMulBiasAddGelu Op Registration Failed!");
  } else {
    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-REG: MatMulBiasAddGelu Op Registration Is Successful!");
  }
  TF_DeleteStatus(status);
}

}  // namespace amd_cpu_plugin

void RegisterZenMatMulOps() {
  amd_cpu_plugin::RegisterZenMatMul();
  amd_cpu_plugin::RegisterZenFusedMatMul();
  amd_cpu_plugin::RegisterZenMatMulBiasAddGelu();
  amd_cpu_plugin::RegisterMatMulBiasAddGelu();
}
