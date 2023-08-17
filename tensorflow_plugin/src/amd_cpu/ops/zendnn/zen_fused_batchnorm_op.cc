/*******************************************************************************
 * Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

// Routine for registering _ZenFusedBatchNorm op.
void RegisterZenFusedBatchNorm() {
  TF_Status* status = TF_NewStatus();

  TF_OpDefinitionBuilder* op_builder =
      TF_NewOpDefinitionBuilder("_ZenFusedBatchNorm");
  TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
  TF_OpDefinitionBuilderAddInput(op_builder, "scale: T");
  TF_OpDefinitionBuilderAddInput(op_builder, "offset: T");
  TF_OpDefinitionBuilderAddInput(op_builder, "mean: T");
  TF_OpDefinitionBuilderAddInput(op_builder, "variance: T");
  TF_OpDefinitionBuilderAddOutput(op_builder, "y: T");
  TF_OpDefinitionBuilderAddOutput(op_builder, "batch_mean: T");
  TF_OpDefinitionBuilderAddOutput(op_builder, "batch_variance: T");
  TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_1: T");
  TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_2: T");
  TF_OpDefinitionBuilderAddAttr(op_builder, "T: { float } = DT_FLOAT");
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                GetConvnetDataFormatAttrString().c_str());
  TF_OpDefinitionBuilderAddAttr(op_builder, "is_eager: bool = false");
  TF_OpDefinitionBuilderAddAttr(op_builder, "reorder_before: bool");
  TF_OpDefinitionBuilderAddAttr(op_builder, "reorder_after: bool");
  TF_OpDefinitionBuilderAddAttr(op_builder, "in_links: int");
  TF_OpDefinitionBuilderAddAttr(op_builder, "out_links: int");
  TF_OpDefinitionBuilderAddAttr(op_builder, "reset: bool");
  TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                "exponential_avg_factor: float = 1.0");
  TF_OpDefinitionBuilderAddAttr(op_builder, "is_training: bool = false");
  TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                  &unknown_shape_fn);

  TF_RegisterOpDefinition(op_builder, status);
  if (TF_OK != TF_GetCode(status)) {
    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-REG: _ZenFusedBatchNorm Op Registration Failed!");
  } else {
    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-REG: _ZenFusedBatchNorm Op Registration Is Successful!");
  }
  TF_DeleteStatus(status);
}

// Routine for registering _ZenFusedBatchNormV2 op.
void RegisterZenFusedBatchNormV2() {
  TF_Status* status = TF_NewStatus();

  TF_OpDefinitionBuilder* op_builder =
      TF_NewOpDefinitionBuilder("_ZenFusedBatchNormV2");
  TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
  TF_OpDefinitionBuilderAddInput(op_builder, "scale: U");
  TF_OpDefinitionBuilderAddInput(op_builder, "offset: U");
  TF_OpDefinitionBuilderAddInput(op_builder, "mean: U");
  TF_OpDefinitionBuilderAddInput(op_builder, "variance: U");
  TF_OpDefinitionBuilderAddOutput(op_builder, "y: T");
  TF_OpDefinitionBuilderAddOutput(op_builder, "batch_mean: U");
  TF_OpDefinitionBuilderAddOutput(op_builder, "batch_variance: U");
  TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_1: U");
  TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_2: U");
  TF_OpDefinitionBuilderAddAttr(op_builder, "T: {float } = DT_FLOAT");
  TF_OpDefinitionBuilderAddAttr(op_builder, "U: {float}");
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                GetConvnetDataFormatAttrString().c_str());
  TF_OpDefinitionBuilderAddAttr(op_builder, "is_eager: bool = false");
  TF_OpDefinitionBuilderAddAttr(op_builder, "reorder_before: bool");
  TF_OpDefinitionBuilderAddAttr(op_builder, "reorder_after: bool");
  TF_OpDefinitionBuilderAddAttr(op_builder, "in_links: int");
  TF_OpDefinitionBuilderAddAttr(op_builder, "out_links: int");
  TF_OpDefinitionBuilderAddAttr(op_builder, "reset: bool");
  TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                "exponential_avg_factor: float = 1.0");
  TF_OpDefinitionBuilderAddAttr(op_builder, "is_training: bool = false");
  TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                  &unknown_shape_fn);

  TF_RegisterOpDefinition(op_builder, status);
  if (TF_OK != TF_GetCode(status)) {
    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-REG: _ZenFusedBatchNormV2 Op Registration Failed!");
  } else {
    zendnnInfo(
        ZENDNN_FWKLOG,
        "ZEN-OP-REG: _ZenFusedBatchNormV2 Op Registration Is Successful!");
  }
  TF_DeleteStatus(status);
}

// Routine for registering _ZenFusedBatchNormV3 op.
void RegisterZenFusedBatchNormV3() {
  TF_Status* status = TF_NewStatus();

  TF_OpDefinitionBuilder* op_builder =
      TF_NewOpDefinitionBuilder("_ZenFusedBatchNormV3");
  TF_OpDefinitionBuilderAddInput(op_builder, "x: T");
  TF_OpDefinitionBuilderAddInput(op_builder, "scale: U");
  TF_OpDefinitionBuilderAddInput(op_builder, "offset: U");
  TF_OpDefinitionBuilderAddInput(op_builder, "mean: U");
  TF_OpDefinitionBuilderAddInput(op_builder, "variance: U");
  TF_OpDefinitionBuilderAddOutput(op_builder, "y: T");
  TF_OpDefinitionBuilderAddOutput(op_builder, "batch_mean: U");
  TF_OpDefinitionBuilderAddOutput(op_builder, "batch_variance: U");
  TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_1: U");
  TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_2: U");
  TF_OpDefinitionBuilderAddOutput(op_builder, "reserve_space_3: U");
  TF_OpDefinitionBuilderAddAttr(op_builder, "T: {float } = DT_FLOAT");
  TF_OpDefinitionBuilderAddAttr(op_builder, "U: {float}");
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                GetConvnetDataFormat2D3DAttrString().c_str());
  TF_OpDefinitionBuilderAddAttr(op_builder, "is_eager: bool = false");
  TF_OpDefinitionBuilderAddAttr(op_builder, "reorder_before: bool");
  TF_OpDefinitionBuilderAddAttr(op_builder, "reorder_after: bool");
  TF_OpDefinitionBuilderAddAttr(op_builder, "in_links: int");
  TF_OpDefinitionBuilderAddAttr(op_builder, "out_links: int");
  TF_OpDefinitionBuilderAddAttr(op_builder, "reset: bool");
  TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                "exponential_avg_factor: float = 1.0");
  TF_OpDefinitionBuilderAddAttr(op_builder, "is_training: bool = false");
  TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                  &unknown_shape_fn);

  TF_RegisterOpDefinition(op_builder, status);
  if (TF_OK != TF_GetCode(status)) {
    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-REG: _ZenFusedBatchNormV3 Op Registration Failed!");
  } else {
    zendnnInfo(
        ZENDNN_FWKLOG,
        "ZEN-OP-REG: _ZenFusedBatchNormV3 Op Registration Is Successful!");
  }
  TF_DeleteStatus(status);
}

}  // namespace amd_cpu_plugin

void RegisterZenFusedBatchNormOps() {
  amd_cpu_plugin::RegisterZenFusedBatchNorm();
  amd_cpu_plugin::RegisterZenFusedBatchNormV2();
  amd_cpu_plugin::RegisterZenFusedBatchNormV3();
}
