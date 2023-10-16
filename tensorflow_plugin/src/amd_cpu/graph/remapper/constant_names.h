/*******************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_GRAPH_REMAPPER_CONSTANT_NAMES_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_GRAPH_REMAPPER_CONSTANT_NAMES_H_

namespace amd_cpu_plugin {
namespace graph {

// Placeholder for pattern matcher.
constexpr char kAny[] = "*";

//  Original TensorFlow op names.
constexpr char kAdd[] = "Add";
constexpr char kAddN[] = "AddN";
constexpr char kAddV2[] = "AddV2";
constexpr char kAssignVariableOp[] = "AssignVariableOp";
constexpr char kBatchMatMulV2[] = "BatchMatMulV2";
constexpr char kBiasAdd[] = "BiasAdd";
constexpr char kBiasAddGrad[] = "BiasAddGrad";
constexpr char kBinaryAdd[] = "BinaryAdd";
constexpr char kBinaryMul[] = "BinaryMul";
constexpr char kCast[] = "Cast";
constexpr char kConcatV2[] = "ConcatV2";
constexpr char kConst[] = "Const";
constexpr char kConv2D[] = "Conv2D";
constexpr char kConv2DBackpropFilter[] = "Conv2DBackpropFilter";
constexpr char kConv3D[] = "Conv3D";
constexpr char kConv3DBackpropFilter[] = "Conv3DBackpropFilter";
constexpr char kConv3DBackpropFilterV2[] = "Conv3DBackpropFilterV2";
constexpr char kDepthwiseConv2dNative[] = "DepthwiseConv2dNative";
constexpr char kDequantize[] = "Dequantize";
constexpr char kFill[] = "Fill";
constexpr char kFusedBatchNormV3[] = "FusedBatchNormV3";
constexpr char kLeakyRelu[] = "LeakyRelu";
constexpr char kMatMul[] = "MatMul";
constexpr char kMean[] = "Mean";
constexpr char kMul[] = "Mul";
constexpr char kPad[] = "Pad";
constexpr char kQuantizeV2[] = "QuantizeV2";
constexpr char kReadVariableOp[] = "ReadVariableOp";
constexpr char kRelu[] = "Relu";
constexpr char kRealDiv[] = "RealDiv";
constexpr char kReshape[] = "Reshape";
constexpr char kResizeNearestNeighbor[] = "ResizeNearestNeighbor";
constexpr char kResizeNearestNeighborGrad[] = "ResizeNearestNeighborGrad";
constexpr char kRsqrt[] = "Rsqrt";
constexpr char kShape[] = "Shape";
constexpr char kSigmoid[] = "Sigmoid";
constexpr char kSlice[] = "Slice";
constexpr char kSoftplus[] = "Softplus";
constexpr char kSplit[] = "Split";
constexpr char kSplitV[] = "SplitV";
constexpr char kSqrt[] = "Sqrt";
constexpr char kSquare[] = "Square";
constexpr char kSquaredDifference[] = "SquaredDifference";
constexpr char kSub[] = "Sub";
constexpr char kTanh[] = "Tanh";

// Fused op names.
constexpr char kFusedConv2D[] = "_FusedConv2D";
constexpr char kFusedDepthwiseConv2dNative[] = "_FusedDepthwiseConv2dNative";
constexpr char kFusedMatMul[] = "_FusedMatMul";

// Misc constant names.
constexpr int kMissingIndex = -1;
constexpr char kDataFormat[] = "data_format";
constexpr char kIsTraining[] = "is_training";

}  // namespace graph
}  // namespace amd_cpu_plugin
#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_GRAPH_REMAPPER_CONSTANT_NAMES_H_
