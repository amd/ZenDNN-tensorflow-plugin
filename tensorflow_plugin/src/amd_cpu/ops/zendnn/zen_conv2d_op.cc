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

// Routine for registering _ZenConv2D op.
void RegisterZenConv2D() {
  TF_Status* status = TF_NewStatus();

  TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder("_ZenConv2D");
  TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
  TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
  TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
  TF_OpDefinitionBuilderAddAttr(op_builder, "T: {float} = DT_FLOAT");
  TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                GetPaddingAttrStringWithExplicit().c_str());
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                GetExplicitPaddingsAttrString().c_str());
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                GetConvnetDataFormatAttrString().c_str());
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                "dilations: list(int) = [1, 1, 1, 1]");
  TF_OpDefinitionBuilderAddAttr(op_builder, "use_cudnn_on_gpu: bool = true");
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
    zendnnInfo(ZENDNN_FWKLOG, "ZEN-OP-REG: _ZenConv2D Op Registration Failed!");
  } else {
    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-REG: _ZenConv2D Op Registration Is Successful!");
  }
  TF_DeleteStatus(status);
}

// Routine for registering _ZenDepthwiseConv2dNative op.
void RegisterZenDepthwiseConv2dNative() {
  TF_Status* status = TF_NewStatus();

  TF_OpDefinitionBuilder* op_builder =
      TF_NewOpDefinitionBuilder("_ZenDepthwiseConv2dNative");
  TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
  TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
  TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
  TF_OpDefinitionBuilderAddAttr(op_builder, "T: {float} = DT_FLOAT");
  TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                GetPaddingAttrStringWithExplicit().c_str());
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                GetExplicitPaddingsAttrString().c_str());
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                GetConvnetDataFormatAttrString().c_str());
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                "dilations: list(int) = [1, 1, 1, 1]");
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
               "ZEN-OP-REG: _ZenDepthwiseConv2dNative Op Registration Failed!");
  } else {
    zendnnInfo(
        ZENDNN_FWKLOG,
        "ZEN-OP-REG: _ZenDepthwiseConv2dNative Op Registration Is Successful!");
  }
  TF_DeleteStatus(status);
}

// Routine for registering _ZenFusedDepthwiseConv2dNative op.
void RegisterZenFusedDepthwiseConv2dNative() {
  TF_Status* status = TF_NewStatus();

  TF_OpDefinitionBuilder* op_builder =
      TF_NewOpDefinitionBuilder("_ZenFusedDepthwiseConv2dNative");
  TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
  TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
  TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
  TF_OpDefinitionBuilderAddAttr(op_builder, "T: {float} = DT_FLOAT");
  TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                GetPaddingAttrStringWithExplicit().c_str());
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                GetExplicitPaddingsAttrString().c_str());
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                GetConvnetDataFormatAttrString().c_str());
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                "dilations: list(int) = [1, 1, 1, 1]");
  TF_OpDefinitionBuilderAddAttr(op_builder, "is_eager: bool = false");
  TF_OpDefinitionBuilderAddAttr(op_builder, "reorder_before: bool");
  TF_OpDefinitionBuilderAddAttr(op_builder, "reorder_after: bool");
  TF_OpDefinitionBuilderAddAttr(op_builder, "in_links: int");
  TF_OpDefinitionBuilderAddAttr(op_builder, "out_links: int");
  TF_OpDefinitionBuilderAddAttr(op_builder, "reset: bool");
  // Fusion specific attributes.
  TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * T");
  TF_OpDefinitionBuilderAddAttr(op_builder, "num_args: int >= 0");
  TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
  TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
  TF_OpDefinitionBuilderAddAttr(op_builder, "leakyrelu_alpha: float = 0.2");
  TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                  &unknown_shape_fn);

  TF_RegisterOpDefinition(op_builder, status);
  if (TF_OK != TF_GetCode(status)) {
    zendnnInfo(
        ZENDNN_FWKLOG,
        "ZEN-OP-REG: _ZenFusedDepthwiseConv2dNative Op Registration Failed!");
  } else {
    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-REG: _ZenFusedDepthwiseConv2dNative Op Registration Is "
               "Successful!");
  }
  TF_DeleteStatus(status);
}

// Routine for registering _ZenFusedConv2D op.
void RegisterZenFusedConv2D() {
  TF_Status* status = TF_NewStatus();

  TF_OpDefinitionBuilder* op_builder =
      TF_NewOpDefinitionBuilder("_ZenFusedConv2D");
  TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
  TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
  TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
  TF_OpDefinitionBuilderAddAttr(op_builder, "T: {float} = DT_FLOAT");
  TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                GetPaddingAttrStringWithExplicit().c_str());
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                GetExplicitPaddingsAttrString().c_str());
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                GetConvnetDataFormatAttrString().c_str());
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                "dilations: list(int) = [1, 1, 1, 1]");
  TF_OpDefinitionBuilderAddAttr(op_builder, "is_eager: bool = false");
  TF_OpDefinitionBuilderAddAttr(op_builder, "reorder_before: bool");
  TF_OpDefinitionBuilderAddAttr(op_builder, "reorder_after: bool");
  TF_OpDefinitionBuilderAddAttr(op_builder, "in_links: int");
  TF_OpDefinitionBuilderAddAttr(op_builder, "out_links: int");
  TF_OpDefinitionBuilderAddAttr(op_builder, "reset: bool");
  // Fusion specific attributes.
  TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * T");
  TF_OpDefinitionBuilderAddAttr(op_builder, "num_args: int >= 0");
  TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
  TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
  TF_OpDefinitionBuilderAddAttr(op_builder, "leakyrelu_alpha: float = 0.2");
  TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                  &unknown_shape_fn);

  TF_RegisterOpDefinition(op_builder, status);
  if (TF_OK != TF_GetCode(status)) {
    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-REG: _ZenFusedConv2D Op Registration Failed!");
  } else {
    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-REG: _ZenFusedConv2D Op Registration Is Successful!");
  }
  TF_DeleteStatus(status);
}

void RegisterZenFusedConv2DSum() {
  TF_Status* status = TF_NewStatus();

  TF_OpDefinitionBuilder* op_builder =
      TF_NewOpDefinitionBuilder("_ZenFusedConv2DSum");
  TF_OpDefinitionBuilderAddInput(op_builder, "input: T");
  TF_OpDefinitionBuilderAddInput(op_builder, "filter: T");
  TF_OpDefinitionBuilderAddOutput(op_builder, "output: T");
  TF_OpDefinitionBuilderAddAttr(op_builder, "T: {float} = DT_FLOAT");
  TF_OpDefinitionBuilderAddAttr(op_builder, "strides: list(int)");
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                GetPaddingAttrStringWithExplicit().c_str());
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                GetExplicitPaddingsAttrString().c_str());
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                GetConvnetDataFormatAttrString().c_str());
  TF_OpDefinitionBuilderAddAttr(op_builder,
                                "dilations: list(int) = [1, 1, 1, 1]");
  TF_OpDefinitionBuilderAddAttr(op_builder, "is_eager: bool = false");
  TF_OpDefinitionBuilderAddAttr(op_builder, "reorder_before: bool");
  TF_OpDefinitionBuilderAddAttr(op_builder, "reorder_after: bool");
  TF_OpDefinitionBuilderAddAttr(op_builder, "in_links: int");
  TF_OpDefinitionBuilderAddAttr(op_builder, "out_links: int");
  TF_OpDefinitionBuilderAddAttr(op_builder, "reset: bool");
  // Fusion specific attributes.
  TF_OpDefinitionBuilderAddInput(op_builder, "args: num_args * T");
  TF_OpDefinitionBuilderAddInput(op_builder, "elementwiseinput: T");
  TF_OpDefinitionBuilderAddAttr(op_builder, "num_args: int >= 0");
  TF_OpDefinitionBuilderAddAttr(op_builder, "fused_ops: list(string) = []");
  TF_OpDefinitionBuilderAddAttr(op_builder, "epsilon: float = 0.0001");
  TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder,
                                                  &unknown_shape_fn);

  TF_RegisterOpDefinition(op_builder, status);
  if (TF_OK != TF_GetCode(status)) {
    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-REG: _ZenFusedConv2DSum Op Registration Failed!!");
  } else {
    zendnnInfo(
        ZENDNN_FWKLOG,
        "ZEN-OP-REG: _ZenFusedConv2DSum Op Registration Is Successful!!");
  }
  TF_DeleteStatus(status);
}

}  // namespace amd_cpu_plugin

void RegisterZenConv2DOps() {
  amd_cpu_plugin::RegisterZenConv2D();
  amd_cpu_plugin::RegisterZenFusedConv2D();
  amd_cpu_plugin::RegisterZenFusedConv2DSum();
  amd_cpu_plugin::RegisterZenDepthwiseConv2dNative();
  amd_cpu_plugin::RegisterZenFusedDepthwiseConv2dNative();
}
