/*******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

namespace amd_cpu_plugin {

// Routine for registering _ZenReshape op.
void RegisterZenReshape() {
  TF_Status* status = TF_NewStatus();

  TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder("_ZenReshape");
  TF_OpDefinitionBuilderAddInput(op_builder, "tensor: T");
  TF_OpDefinitionBuilderAddInput(op_builder, "shape: Tshape");
  TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
  TF_OpDefinitionBuilderAddAttr(op_builder, "T: type");
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                "Tshape: {int32, int64} = DT_INT32");
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
               "ZEN-OP-REG: _ZenReshape Op Registration Failed!");
  } else {
    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-REG: _ZenReshape Op Registration Is Successful!");
  }
  TF_DeleteStatus(status);
}

}  // namespace amd_cpu_plugin

// Routine for registering ZenReshape related ops.
void RegisterZenReshapeOp() { amd_cpu_plugin::RegisterZenReshape(); }