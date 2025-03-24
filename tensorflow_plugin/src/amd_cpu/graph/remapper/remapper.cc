/*******************************************************************************
 * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 ******************************************************************************/

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow_plugin/src/amd_cpu/graph/remapper/remapper.h"

#include <utility>

#include "tensorflow_plugin/src/amd_cpu/graph/remapper/constant_names.h"
#include "tensorflow_plugin/src/amd_cpu/graph/utils/graph_common_utils.h"
#include "tensorflow_plugin/src/amd_cpu/graph/utils/layout_utils.h"
#include "tensorflow_plugin/src/amd_cpu/graph/utils/op_types.h"
#include "tensorflow_plugin/src/amd_cpu/graph/utils/pattern_utils.h"
#include "tensorflow_plugin/src/amd_cpu/graph/utils/symbolic_shapes.h"

namespace amd_cpu_plugin {
namespace graph {

bool HasDataType(const NodeDef* node, const DataType& expected,
                 const string& type_attr) {
  DataType dtype = GetDataTypeFromAttr(*node, type_attr);
  return dtype == expected;
}

bool IsSupportedActivation(const NodeDef& node) {
  bool is_default_supported = IsRelu(node) || IsRelu6(node) || IsElu(node) ||
                              IsLeakyRelu(node) || IsSigmoid(node);
  return is_default_supported;
}

void SetFusedOpAttributes(NodeDef* fused,
                          const absl::Span<const absl::string_view> fused_ops,
                          int num_args = 1, float epsilon = 0.0) {
  auto* attr = fused->mutable_attr();
  SetAttrValue(fused_ops, &(*attr)["fused_ops"]);
  SetAttrValue(num_args, &(*attr)["num_args"]);
  SetAttrValue(epsilon, &(*attr)["epsilon"]);
}

// Helper function to set fused op attributes with activation.
// `fused_ops` should not contain `activation`, it will add activation
// in this function.
void SetFusedOpAttributesWithActivation(
    NodeDef* fused, const NodeDef* activation,
    std::vector<absl::string_view> fused_ops, int num_args = 1) {
  // Handle special activation.
  if (activation != nullptr) {
    auto& activation_attr = activation->attr();

    if (IsLeakyRelu(*activation)) {
      AddNodeAttr("leakyrelu_alpha", activation_attr.at("alpha"), fused);
      fused_ops.push_back(activation->op());
    } else if (IsGelu(*activation)) {
      fused_ops.push_back(activation_attr.at("approximate").b()
                              ? "GeluApproximate"
                              : "GeluExact");
    } else {
      fused_ops.push_back(activation->op());
    }
  }

  SetFusedOpAttributes(fused, fused_ops, num_args);
}

Status GetTensorFromConstant(const NodeDef* node_def, Tensor* dst) {
  if (!dst->FromProto(node_def->attr().at("value").tensor())) {
    TF_CHECK_OK(errors::InvalidArgument(
        "Could not construct Tensor from TensorProto in node: ",
        node_def->name()));
  }

  return OkStatus();
}

namespace {

// FusedBatchNorm with activation.
struct FusedBatchNormEx {
  FusedBatchNormEx() = default;

  int fused_batch_norm = kMissingIndex;
  int activation = kMissingIndex;
};

// Contraction node followed by a BiasAdd.
struct ContractionWithBiasAdd {
  ContractionWithBiasAdd() = default;
  ContractionWithBiasAdd(int contraction, int bias_add, int bias_port)
      : contraction(contraction), bias_add(bias_add), bias_port(bias_port) {}

  int contraction = kMissingIndex;
  int bias_add = kMissingIndex;
  int bias_port = kMissingIndex;
};

// Contraction node followed by a BiasAdd and Activation.
struct ContractionWithBiasAddAndActivation {
  ContractionWithBiasAddAndActivation() = default;
  ContractionWithBiasAddAndActivation(int contraction, int bias_add,
                                      int activation, int bias_port)
      : contraction(contraction),
        bias_add(bias_add),
        activation(activation),
        bias_port(bias_port) {}

  int contraction = kMissingIndex;
  int bias_add = kMissingIndex;
  int activation = kMissingIndex;
  int bias_port = kMissingIndex;
};

// Contraction node followed by a FusedBatchNorm.
struct ContractionWithBatchNorm {
  ContractionWithBatchNorm() = default;
  ContractionWithBatchNorm(int contraction, int fused_batch_norm,
                           float epsilon = 0.0)
      : contraction(contraction),
        fused_batch_norm(fused_batch_norm),
        epsilon(epsilon) {}

  int contraction = kMissingIndex;
  int fused_batch_norm = kMissingIndex;
  float epsilon = 0.0;
};

// Contraction node followed by a FusedBatchNorm and Activation.
struct ContractionWithBatchNormAndActivation {
  ContractionWithBatchNormAndActivation() = default;
  ContractionWithBatchNormAndActivation(int contraction, int fused_batch_norm,
                                        int activation, float epsilon = 0.0)
      : contraction(contraction),
        fused_batch_norm(fused_batch_norm),
        activation(activation),
        epsilon(epsilon) {}

  int contraction = kMissingIndex;
  int fused_batch_norm = kMissingIndex;
  int activation = kMissingIndex;
  float epsilon = 0.0;
};

// Contraction node followed by a BiasAdd and Add.
struct ContractionWithBiasAddAndAdd {
  ContractionWithBiasAddAndAdd() = default;
  ContractionWithBiasAddAndAdd(int contraction, int bias_add, int add,
                               int port_id, int bias_port)
      : contraction(contraction),
        bias_add(bias_add),
        add(add),
        port_id(port_id),
        bias_port(bias_port) {}

  int contraction = kMissingIndex;
  int bias_add = kMissingIndex;
  int add = kMissingIndex;
  int port_id = 0;
  int bias_port = kMissingIndex;
};

// Contraction node followed by a BiasAdd, Add and Relu.
struct ContractionWithBiasAndAddActivation {
  ContractionWithBiasAndAddActivation() = default;
  ContractionWithBiasAndAddActivation(int contraction, int bias_add, int add,
                                      int port_id, int activation,
                                      int bias_port)
      : contraction(contraction),
        bias_add(bias_add),
        add(add),
        port_id(port_id),
        activation(activation),
        bias_port(bias_port) {}

  int contraction = kMissingIndex;
  int bias_add = kMissingIndex;
  int add = kMissingIndex;
  int port_id = 0;
  int activation = kMissingIndex;
  int bias_port = kMissingIndex;
};

// MatMul node followed by a Activation.
struct ContractionWithActivation {
  ContractionWithActivation() = default;
  ContractionWithActivation(int contraction, int activation)
      : contraction(contraction), activation(activation) {}

  int contraction = kMissingIndex;
  int activation = kMissingIndex;
};

// Pad with `VALID` and 'EXPLICIT' padding followed by Depthwise/_Fused(Conv2D).
// Only `Pad` is supported rather than PadV2/MirrorPad.
struct PadWithContraction {
  PadWithContraction() = default;
  PadWithContraction(int pad, int contraction)
      : pad(pad), contraction(contraction) {}

  int pad = kMissingIndex;
  int contraction = kMissingIndex;
};

struct KerasDenseLayerFwd {
  KerasDenseLayerFwd() = default;
  KerasDenseLayerFwd(int matmul, int reshape, int bias, int activation)
      : matmul(matmul), reshape(reshape), bias(bias), activation(activation) {}

  int matmul = kMissingIndex;
  int reshape = kMissingIndex;
  int bias = kMissingIndex;
  int activation = kMissingIndex;
};

// BatchMatMul + Mul fusion.
struct ContractionWithMul {
  ContractionWithMul() = default;
  ContractionWithMul(int contraction, int mul, int scalar)
      : contraction(contraction), mul(mul), scalar(scalar) {}

  int contraction = kMissingIndex;
  int mul = kMissingIndex;
  int scalar = kMissingIndex;
};

bool IsAddWithNoBroadcast(const RemapperContext& ctx, const NodeDef& node) {
  if (!IsAdd(node)) return false;

  // Check if this is case of broadcasting - Add node supports broadcasting.
  std::vector<OpInfo_TensorProperties> props;
  TF_ABORT_IF_ERROR(
      ctx.graph_properties.GetInputProperties(node.name(), &props));
  if (props.size() == 2 &&
      ShapesSymbolicallyEqual(props[0].shape(), props[1].shape())) {
    return true;
  }
  return false;
}

// Generic function to check contraction kernel.
bool IsConvOrMatMul(const NodeDef& node) {
  return IsConv3D(node) || IsConv2D(node) || IsDepthwiseConv2dNative(node) ||
         IsMatMul(node);
}

// Returns true if one input to Add is Conv2D/3D or DepthwiseConv2dNative or
// MatMul, and the other input is semantically equivalent to BiasAdd.
bool IsBiasSemanticAdd(const RemapperContext& ctx,
                       const utils::MutableNodeView& node_view,
                       int* bias_port) {
  const auto* node_def = node_view.node();
  if (!IsAdd(*node_def) || node_view.NumRegularFanins() != 2) return false;

  std::vector<OpInfo_TensorProperties> props;
  TF_ABORT_IF_ERROR(
      ctx.graph_properties.GetInputProperties(node_def->name(), &props));

  if (props.size() < 2) return false;

  const auto& regular_fanin_0 = node_view.GetRegularFanin(0);
  const auto* node_view_0 = regular_fanin_0.node_view();
  const auto* node_def_0 = node_view_0->node();
  const auto& regular_fanin_1 = node_view.GetRegularFanin(1);
  const auto* node_view_1 = regular_fanin_1.node_view();
  const auto* node_def_1 = node_view_1->node();

  // Currently supported data formats are NHWC.
  auto is_channel_last_format = [](const NodeDef& node) -> bool {
    if (node.attr().contains("data_format")) {
      const string data_format = node.attr().at("data_format").s();
      return (data_format == "NHWC");
    }
    return true;
  };

  if (IsConvOrMatMul(*node_def_0) && is_channel_last_format(*node_def_0)) {
    *bias_port = 1;
  } else if (IsConvOrMatMul(*node_def_1) &&
             is_channel_last_format(*node_def_1)) {
    *bias_port = 0;
  } else {
    return false;
  }

  const TensorShapeProto& contraction_shape = props[1 - *bias_port].shape();
  const TensorShapeProto& bias_shape = props[*bias_port].shape();

  if (contraction_shape.unknown_rank() || bias_shape.unknown_rank() ||
      contraction_shape.dim_size() < 1 || bias_shape.dim_size() < 1 ||
      IsUnknown(contraction_shape.dim(contraction_shape.dim_size() - 1)) ||
      IsUnknown(bias_shape.dim(bias_shape.dim_size() - 1)))
    return false;

  // Helper function to check Add/AddV2 could be replaced with BiasAdd.
  const auto is_supported_shape =
      [&](const TensorShapeProto& shape,
          const TensorShapeProto& bcast_shape) -> bool {
    int conv_channel_dim;
    conv_channel_dim = shape.dim(shape.dim_size() - 1).size();

    if (shape.dim_size() == 4 && bcast_shape.dim_size() > 4) return false;
    if (shape.dim_size() == 5 && bcast_shape.dim_size() > 5) return false;

    if (shape.dim_size() < 2) return false;
    // Check that the conv node's channel dim is equal to the 1-dim add node's
    // dim.
    if (conv_channel_dim != bcast_shape.dim(bcast_shape.dim_size() - 1).size())
      return false;

    // Check that add nodes dims are all 1's except the channel dim.
    for (int i = 0; i < bcast_shape.dim_size() - 1; i++) {
      if (1 != bcast_shape.dim(i).size()) return false;
    }
    return true;
  };

  if (ShapesSymbolicallyEqual(contraction_shape, bias_shape) ||
      !ShapesBroadcastable(contraction_shape, bias_shape))
    return false;

  return is_supported_shape(contraction_shape, bias_shape);
}

// Returns 0: left input scalar, 1: right input scalar, -1: no scalar inputs.
int GetMulScalarInputIndex(const RemapperContext& ctx,
                           const NodeDef& node_def) {
  std::vector<OpInfo_TensorProperties> props;
  TF_ABORT_IF_ERROR(
      ctx.graph_properties.GetInputProperties(node_def.name(), &props));
  if (props.size() != 2) return -1;

  bool left_is_scalar = IsScalar(props[0].shape());
  bool right_is_scalar = IsScalar(props[1].shape());
  if (left_is_scalar) {
    return 0;
  } else if (right_is_scalar) {
    return 1;
  } else {
    return -1;
  }
}

// The function to set shapes is used in TF Proper's fused op creation
// extensively, but is not necessary in fused op creation in plugin except for
// BatchNorm fusions.
// TODO(plugin) : Validate the necessity of the same.
void AddInputShapesAttr(const RemapperContext& ctx, int node_index) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  NodeDef* node_def = node_view->node();

  AttrValue attr_input_shape;
  std::vector<OpInfo_TensorProperties> tensor_properties;
  TF_ABORT_IF_ERROR(ctx.graph_properties.GetInputProperties(
      node_def->name(), &tensor_properties));
  for (const auto& tensor_property : tensor_properties) {
    TensorShapeProto* proto = attr_input_shape.mutable_list()->add_shape();
    *proto = tensor_property.shape();
  }

  // TODO(plugin): Validate if "input_shapes" is necessary for ZenDNN ops.
  if (!tensor_properties.empty()) {
    auto* attr = node_def->mutable_attr();
    SetAttrValue(attr_input_shape, &(*attr)["_input_shapes"]);
  }
}

bool FindKerasDenseLayerFwd(const RemapperContext& ctx, int node_index,
                            KerasDenseLayerFwd* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  // Root of the pattern must be a Reshape.
  // Find reshape + biasadd + activation.
  int bias_index = kMissingIndex;
  int activation_index = kMissingIndex;
  const auto* reshape = node_view->node();
  if (!reshape || !IsReshape(*reshape) || HasControlFaninOrFanout(*node_view) ||
      IsInPreserveSet(ctx, node_view->node()))
    return false;

  if (node_view->NumRegularFanouts() != 1) return false;
  const auto& reshape_fanout_0 = node_view->GetRegularFanouts()[0];
  if (reshape_fanout_0.size() != 1) return false;

  const auto* biasadd = reshape_fanout_0[0].node_view();
  if (!IsBiasAdd(*biasadd->node()) || HasControlFaninOrFanout(*biasadd) ||
      IsInPreserveSet(ctx, biasadd->node()))
    return false;
  bias_index = biasadd->node_index();
  if (biasadd->NumRegularFanouts() == 1) {
    const auto& biasadd_fanout_0 = biasadd->GetRegularFanouts()[0];
    if (biasadd_fanout_0.size() == 1) {
      const auto* relu = biasadd_fanout_0[0].node_view();
      if (IsSupportedActivation(*relu->node()) &&
          !HasControlFaninOrFanout(*relu) &&
          !IsInPreserveSet(ctx, relu->node())) {
        activation_index = relu->node_index();
      }
    }
  }

  int bias_dim = 0;
  int weight_dim = 0;
  if (biasadd->NumRegularFanins() != 2) return false;
  auto* readvariable = biasadd->GetRegularFanin(1).node_view();
  // If pb is frozen.
  if (IsConstant(*readvariable->node())) {
    const TensorProto& bias_val =
        readvariable->node()->attr().at("value").tensor();
    const TensorShape bias_shape(bias_val.tensor_shape());
    bias_dim = bias_shape.num_elements();
  }

  // _Arg -> ReadVariableOp -> (Cast) ->BiasAdd.
  if (bias_dim == 0) {
    if (IsCast(*readvariable->node())) {
      readvariable = readvariable->GetRegularFanin(0).node_view();
    }
    if (!IsReadVariableOp(*readvariable->node())) return false;
    const auto* arg_bias = readvariable->GetRegularFanin(0).node_view()->node();

    if (IsArg(*arg_bias)) {
      const AttrValue attr_bshape = arg_bias->attr().at("_handle_shapes");
      if (attr_bshape.list().shape().empty()) return false;
      const TensorShapeProto& bshape_proto = attr_bshape.list().shape(0);
      if (bshape_proto.unknown_rank()) return false;
      bias_dim = TensorShape(bshape_proto).dim_size(0);
    } else if (IsVarHandle(*arg_bias)) {
      const AttrValue attr_bshape = arg_bias->attr().at("shape");

      const TensorShapeProto& bshape_proto = attr_bshape.shape();
      if (bshape_proto.unknown_rank() ||
          IsUnknown(bshape_proto.dim(bshape_proto.dim_size() - 1)))
        return false;
      bias_dim = TensorShape(bshape_proto).dim_size(0);
    } else {
      return false;
    }
  }
  // Arg -> ReadVariableOp -> (Cast) ->MatMul -> reshape.
  if (node_view->NumRegularFanins() != 2) return false;
  const auto* matmul = node_view->GetRegularFanin(0).node_view();
  if (!IsMatMul(*matmul->node())) return false;
  auto* readvariable2 = matmul->GetRegularFanin(1).node_view();

  // If pb is frozen.
  if (IsConstant(*readvariable2->node())) {
    const TensorProto& weight_val =
        readvariable2->node()->attr().at("value").tensor();
    const TensorShape weight_shape(weight_val.tensor_shape());
    weight_dim = weight_shape.dim_size(1);
  }

  if (weight_dim == 0) {
    if (IsCast(*readvariable2->node())) {
      readvariable2 = readvariable2->GetRegularFanin(0).node_view();
    }
    if (!IsReadVariableOp(*readvariable2->node())) return false;
    const auto* arg_weight =
        readvariable2->GetRegularFanin(0).node_view()->node();
    if (IsArg(*arg_weight)) {
      const AttrValue attr_wshape = arg_weight->attr().at("_handle_shapes");
      if (attr_wshape.list().shape().empty()) return false;
      const TensorShapeProto& wshape_proto = attr_wshape.list().shape(0);
      if (!Is2D(wshape_proto)) return false;
      weight_dim = TensorShape(wshape_proto).dim_size(1);
    } else if (IsVarHandle(*arg_weight)) {
      const AttrValue attr_wshape = arg_weight->attr().at("shape");
      const TensorShapeProto& wshape_proto = attr_wshape.shape();
      if (!Is2D(wshape_proto)) return false;
      if (IsUnknown(wshape_proto.dim(wshape_proto.dim_size() - 1)))
        return false;
      weight_dim = TensorShape(wshape_proto).dim_size(1);
    } else {
      return false;
    }
  }

  if (bias_dim != weight_dim) return false;

  const KerasDenseLayerFwd pattern{matmul->node_index(),
                                   node_view->node_index(), bias_index,
                                   activation_index};
  *matched = pattern;
  return true;
}

bool FindContractionWithBias(const RemapperContext& ctx, int node_index,
                             ContractionWithBiasAdd* matched,
                             bool check_device_compatible = true) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);

  // Verify the output node has control fanin edge or not.
  if (HasControlFanin(*node_view)) return false;

  const auto* node_def = node_view->node();
  int bias_port = 1;
  if (!IsBiasAdd(*node_def) && !IsBiasSemanticAdd(ctx, *node_view, &bias_port))
    return false;

  // Input to the BiasAdd must be a Conv2D or a MatMul.
  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(1 - bias_port);
  const auto* contraction_node_view = regular_fanin_0.node_view();
  const auto* contraction_node_def = contraction_node_view->node();

  // Verify the input node has a control fanout edge or not.
  if (HasControlFanout(*contraction_node_view)) return false;

  // Conv, MatMul or DepthwiseConv2D.
  bool is_contraction = IsConvOrMatMul(*contraction_node_def) ||
                        IsAnyBatchMatMul(*contraction_node_def);

  if (!is_contraction || !HaveSameDataType(node_def, contraction_node_def) ||
      !HasAtMostOneFanoutAtPort0(*contraction_node_view) ||
      IsInPreserveSet(ctx, contraction_node_def))
    return false;

  const ContractionWithBiasAdd pattern{contraction_node_view->node_index(),
                                       node_index, bias_port};
  // We successfully found a {Conv2D, MatMul}+BiasAdd pattern.
  *matched = pattern;
  return true;
}

// As AddN has multiple inputs, this function tries to find Conv2D + Bias
// pattern in specific input port.
bool FindContractionWithBiasInPort(const RemapperContext& ctx,
                                   const utils::MutableNodeView& add_node_view,
                                   const NodeDef& add_node_def, int port_id,
                                   ContractionWithBiasAdd* base) {
  // Input to AddN must match ContractionWithBiasAdd pattern.
  if (add_node_view.NumRegularFanins() < port_id + 1) return false;
  const auto& bias_add_node_view =
      add_node_view.GetRegularFanin(port_id).node_view();
  if (bias_add_node_view == nullptr) return false;
  const auto* bias_add_node_def = bias_add_node_view->node();

  if (!FindContractionWithBias(ctx, bias_add_node_view->node_index(), base,
                               /*check_device_compatible=*/false))
    return false;
  if (!HasAtMostOneFanoutAtPort0(*bias_add_node_view) ||
      !HaveSameDataType(&add_node_def, bias_add_node_def) ||
      IsInPreserveSet(ctx, bias_add_node_def))
    return false;
  return true;
}

bool FindContractionWithBiasAddAndAdd(const RemapperContext& ctx,
                                      const int node_index,
                                      ContractionWithBiasAddAndAdd* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  // Fusion with AddN is supported only when it has two inputs.
  if (HasControlFaninOrFanout(*node_view) || node_view->NumRegularFanins() != 2)
    return false;

  // Root of the pattern must be a AddN or Add with same input shapes
  // (no broadcasting).
  const auto* node_def = node_view->node();

  if (!IsAddN(*node_def) && !IsAddWithNoBroadcast(ctx, *node_def)) return false;

  if (!HasDataType(node_def, DT_FLOAT) && !HasDataType(node_def, DT_BFLOAT16))
    return false;

  ContractionWithBiasAdd base;
  matched->port_id = 0;

  // Find the conv+bias pattern in specific port.
  if (!FindContractionWithBiasInPort(ctx, *node_view, *node_def,
                                     matched->port_id, &base)) {
    matched->port_id = 1;
    if (!FindContractionWithBiasInPort(ctx, *node_view, *node_def,
                                       matched->port_id, &base)) {
      return false;
    }
  }

  // We do not yet have support for DepthWiseConv2D fusion.
  const auto* contraction_def =
      ctx.graph_view.GetNode(base.contraction)->node();
  if (IsDepthwiseConv2dNative(*contraction_def)) return false;

  // We successfully found a Conv2D+BiasAdd+{AddN,Add} pattern.
  matched->contraction = base.contraction;
  matched->bias_add = base.bias_add;
  matched->bias_port = base.bias_port;
  matched->add = node_view->node_index();

  return true;
}

bool FindContractionWithActivation(const RemapperContext& ctx, int node_index,
                                   ContractionWithActivation* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  // Root of the pattern must be an activation node.
  const auto* node_def = node_view->node();
  if (node_def == nullptr) return false;
  if (!IsRelu(*node_def)) return false;

  // Verify the output node has control fanin edge or not.
  if (HasControlFanin(*node_view)) return false;

  // Input to the Relu must be a MatMul.
  // We have not yet encountered other Contraction + Activation patterns.
  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* contraction_node_view = regular_fanin_0.node_view();
  const auto* contraction_node_def = contraction_node_view->node();

  bool is_matmul = IsMatMul(*contraction_node_def);

  // Verify the input node has a control fanout edge or not.
  if (HasControlFanout(*contraction_node_view)) return false;

  if (!is_matmul || !HaveSameDataType(node_def, contraction_node_def) ||
      !HasAtMostOneFanoutAtPort0(*contraction_node_view) ||
      IsInPreserveSet(ctx, contraction_node_def))
    return false;

  const ContractionWithActivation pattern{contraction_node_view->node_index(),
                                          node_index};
  // We successfully found a Matmul + Relu pattern.
  *matched = pattern;
  return true;
}

bool FindContractionWithSigmoid(const RemapperContext& ctx, int node_index,
                                ContractionWithActivation* matched) {
  // Get the node at the given index.
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();
  if (node_def == nullptr) return false;

  // The root of the pattern must be a Sigmoid node.
  if (!IsSigmoid(*node_def)) return false;

  // Sigmoid should have no control inputs.
  if (HasControlFanin(*node_view)) return false;

  // Ensure Sigmoid has at least one regular input.
  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* contraction_node_view = regular_fanin_0.node_view();
  const auto* contraction_node_def = contraction_node_view->node();

  // Verify the input is _FusedMatMul as we have encountered _FuseMatMul +
  // Sigmoid.
  if (contraction_node_def->op() != "_FusedMatMul") return false;

  // Check if _FusedMatMul contains only BiasAdd.
  auto fused_ops = contraction_node_def->attr().at("fused_ops").list().s();
  if (fused_ops.size() != 1 || fused_ops.at(0) != "BiasAdd") return false;

  // Additional constraints for fusion.
  if (HasControlFanout(*contraction_node_view) ||  // No control outputs.
      !HaveSameDataType(node_def,
                        contraction_node_def) ||  // Matching data types.
      !HasAtMostOneFanoutAtPort0(
          *contraction_node_view) ||                 // At most one fanout.
      IsInPreserveSet(ctx, contraction_node_def)) {  // Not preserved.
    return false;
  }

  // Pattern matched; store the node indices.
  const ContractionWithActivation pattern{contraction_node_view->node_index(),
                                          node_index};
  *matched = pattern;
  return true;
}

bool FindContractionWithBiasAndActivation(
    const RemapperContext& ctx, int node_index,
    ContractionWithBiasAddAndActivation* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  // Root of the pattern must be an activation node.
  const auto* node_def = node_view->node();
  if (node_def == nullptr) return false;
  // TODO: Add check for Convolution + BiasAdd + Sigmoid fusion.
  if (!IsSupportedActivation(*node_def)) return false;

  // Verify the output node has control fanin edge or not.
  if (HasControlFanin(*node_view)) return false;

  // And input to the activation node must match ContractionWithBiasAdd pattern.
  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* bias_add_node_view = regular_fanin_0.node_view();
  const auto* bias_add_node_def = bias_add_node_view->node();

  ContractionWithBiasAdd base;
  if (!FindContractionWithBias(ctx, bias_add_node_view->node_index(), &base,
                               /*check_device_compatible=*/false) ||
      !HasAtMostOneFanoutAtPort0(*bias_add_node_view) ||
      (!HaveSameDataType(node_def, bias_add_node_def) &&
       !(GetDataTypeFromAttr(*node_def, "T") == DT_FLOAT)) ||
      IsInPreserveSet(ctx, bias_add_node_def))
    return false;

  // TODO(plugin): TF Proper doesn't have MatMul + LeakyRelu fusion, remove this
  // limitation once it's supported.
  const auto* contraction_node_view = ctx.graph_view.GetNode(base.contraction);
  const auto* contraction_def = contraction_node_view->node();

  // We have not encountered any other Contraction + BiasAdd + {Sigmoid}
  // pattern.
  if (IsSigmoid(*node_def) && !IsMatMul(*contraction_def)) return false;

  // Verify the inter node has control fanin&fanout or not.
  if (HasControlFaninOrFanout(*bias_add_node_view)) {
    return false;
  }

  // TODO(plugin): ZenDNN does not support double dtype currently.
  if (HasDataType(contraction_def, DT_DOUBLE)) return false;
  if (IsLeakyRelu(*node_def) && IsMatMul(*contraction_def)) return false;

  const ContractionWithBiasAddAndActivation pattern{
      base.contraction, base.bias_add, node_index, base.bias_port};

  // Verify the input node has a control fanout edge or not.
  if (HasControlFanout(*contraction_node_view)) return false;

  // We successfully found a {Conv2D, MatMul}+BiasAdd+Activation pattern.
  *matched = pattern;

  return true;
}

bool FindFusedBatchNormEx(const RemapperContext& ctx, int node_index,
                          FusedBatchNormEx* matched) {
  // Root of the pattern must be a Relu.
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();
  if (!IsRelu(*node_def)) return false;

  // Returns true iff the node is a compatible FusedBatchNorm node.
  const auto valid_batch_norm =
      [&](const utils::MutableNodeView& fused_batch_norm) -> bool {
    const auto* fused_batch_norm_node_def = fused_batch_norm.node();
    if (!IsFusedBatchNorm(*fused_batch_norm_node_def)) return false;

    DataType t_dtype = GetDataTypeFromAttr(*fused_batch_norm_node_def, "T");

    // CPU supports float.
    if (t_dtype != DT_FLOAT) return false;

    string data_format;
    if (!GetNodeAttr(*fused_batch_norm_node_def, kDataFormat, &data_format)
             .ok())
      return false;
    if (data_format != "NHWC" && data_format != "NCHW") return false;

    // FusedBatchNormV2 and V3 have an extra type parameter.
    if ((fused_batch_norm_node_def->op() != "FusedBatchNorm") &&
        !HasDataType(fused_batch_norm_node_def, DT_FLOAT, "U"))
      return false;

    // Check that only one node consumes the 0-th output of a FusedBatchNorm.
    if (HasControlFaninOrFanout(fused_batch_norm) ||
        !HasAtMostOneFanoutAtPort0(fused_batch_norm) ||
        IsInPreserveSet(ctx, fused_batch_norm_node_def))
      return false;

    return true;
  };

  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* relu_fanin_0_node_view = regular_fanin_0.node_view();

  // Input to a Relu can be a FusedBatchNorm.
  if (valid_batch_norm(*relu_fanin_0_node_view)) {
    matched->activation = node_index;
    matched->fused_batch_norm = regular_fanin_0.node_index();
    return true;
  }

  return false;
}

bool FindConv2DWithBatchNorm(const RemapperContext& ctx, int node_index,
                             ContractionWithBatchNorm* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();

  // Fusion type is not supported for BF16 path because of the lack of API
  // support from ZenDNN.
  // TODO(zendnn) : Add the API support from ZenDNN library.
  if (HasDataType(node_def, DT_BFLOAT16)) return false;

  // Root of the pattern must be a FusedBatchNorm.
  if (!IsFusedBatchNorm(*node_def)) return false;

  // Check that batch normalization is in inference mode.
  const auto* training_attr = node_view->GetAttr(kIsTraining);
  if (training_attr != nullptr && training_attr->b()) return false;

  // Check that only 0th output is consumed by other nodes.
  // TODO(plugin): Forward controls for patterns with control dependencies.
  if (HasControlFaninOrFanout(*node_view) ||
      !node_view->GetRegularFanout(1).empty() ||  // batch_mean
      !node_view->GetRegularFanout(2).empty() ||  // batch_variance
      !node_view->GetRegularFanout(3).empty() ||  // reserve_space_1
      !node_view->GetRegularFanout(4).empty())    // reserve_space_2
    return false;

  // Input to the FusedBatchNorm must be a Conv2D.
  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* conv2d_node_view = regular_fanin_0.node_view();
  const auto* conv2d_node_def = conv2d_node_view->node();

  // TODO(plugin) : Verify and add checks for CPU compatible data type and
  // data format if necessary.
  if (!IsConv2D(*conv2d_node_def) || !NodeIsOnCpu(conv2d_node_def) ||
      !HaveSameDataType(node_def, conv2d_node_def) ||
      HasControlFaninOrFanout(*conv2d_node_view) ||
      !HasAtMostOneFanoutAtPort0(*conv2d_node_view) ||
      IsInPreserveSet(ctx, conv2d_node_def))
    return false;

  // We successfully found a Conv2D+FusedBatchNorm pattern.
  matched->contraction = conv2d_node_view->node_index();
  matched->fused_batch_norm = node_index;
  if (!TryGetNodeAttr(*node_def, "epsilon", &matched->epsilon)) return false;

  return true;
}

bool FindConv2DWithBatchNormAndActivation(
    const RemapperContext& ctx, int node_index,
    ContractionWithBatchNormAndActivation* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();

  // TODO(plugin): Forward controls for patterns with control dependencies.
  if (HasControlFaninOrFanout(*node_view)) return false;

  // Root of the pattern must be an activation node.
  if (!IsSupportedActivation(*node_def)) return false;

  // And input to the activation node must match Conv2DWithBatchNorm pattern.
  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* batch_norm_node_view = regular_fanin_0.node_view();

  ContractionWithBatchNorm base;
  if (!FindConv2DWithBatchNorm(ctx, batch_norm_node_view->node_index(), &base))
    return false;

  const auto* fused_batch_norm_node_view =
      ctx.graph_view.GetNode(base.fused_batch_norm);
  const auto* fused_batch_norm_node_def = fused_batch_norm_node_view->node();
  if (!HasAtMostOneFanoutAtPort0(*fused_batch_norm_node_view) ||
      !HaveSameDataType(node_def, fused_batch_norm_node_def) ||
      IsInPreserveSet(ctx, fused_batch_norm_node_def))
    return false;

  // We successfully found a Conv2D+FusedBatchNorm+Activation pattern.
  matched->contraction = base.contraction;
  matched->fused_batch_norm = base.fused_batch_norm;
  matched->activation = node_index;
  matched->epsilon = base.epsilon;

  return true;
}

bool FindContractionWithBiasAndAddActivation(
    const RemapperContext& ctx, int node_index,
    ContractionWithBiasAndAddActivation* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);

  if (HasControlFaninOrFanout(*node_view)) return false;
  // Root of the pattern must be an activation node.
  const auto* node_def = node_view->node();
  if (node_def == nullptr) return false;
  if (!IsSupportedActivation(*node_def)) return false;

  // ZenDNN activation op only supports float and bfloat16 on CPU.
  if (!HasDataType(node_def, DT_FLOAT) && !HasDataType(node_def, DT_BFLOAT16))
    return false;

  // And input to activation must match ContractionWithBiasAddAndAdd pattern.
  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* add_node_view = regular_fanin_0.node_view();
  const auto* add_node_def = add_node_view->node();

  ContractionWithBiasAddAndAdd base;
  if (!FindContractionWithBiasAddAndAdd(ctx, add_node_view->node_index(),
                                        &base) ||
      !HasAtMostOneFanoutAtPort0(*add_node_view) ||
      !HaveSameDataType(node_def, add_node_def) ||
      IsInPreserveSet(ctx, add_node_def)) {
    return false;
  }

  // TODO(plugin): Public TF doesn't have MatMul + LeakyRelu fusion, remove this
  // limitation once it's supported.
  const auto* contraction_def =
      ctx.graph_view.GetNode(base.contraction)->node();
  if (IsLeakyRelu(*node_def) && IsMatMul(*contraction_def)) return false;

  // We successfully found a Conv2D+BiasAdd+AddN+activation pattern.
  const ContractionWithBiasAndAddActivation pattern{
      base.contraction, base.bias_add, base.add,
      base.port_id,     node_index,    base.bias_port};
  *matched = pattern;

  return true;
}

bool FindPadWithContraction(const RemapperContext& ctx, int node_index,
                            PadWithContraction* matched,
                            bool check_device_compatible = true) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  // Root of the pattern must be a Conv or FusedConv.
  if (HasControlFaninOrFanout(*node_view)) return false;

  // Root node must be (_Fused)Conv2D/(_Fused)DepthwiseConv2dNative.
  const auto* node_def = node_view->node();
  const bool is_ok = IsConv2D(*node_def) || node_def->op() == kFusedConv2D ||
                     IsDepthwiseConv2dNative(*node_def) ||
                     node_def->op() == kFusedDepthwiseConv2dNative;
  if (!is_ok) {
    return false;
  }

  // Input to the contraction must be Pad.
  if (node_view->NumRegularFanins() < 1) return false;
  const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
  const auto* pad_node_view = regular_fanin_0.node_view();
  const auto* pad_node_def = pad_node_view->node();

  // Only Pad is allowed, PadV2 will be prevented.
  if (pad_node_def->op() != "Pad") return false;

  // Only fuse contraction with `VALID` and 'EXPLICIT' padding.
  // TODO(plugin): Support more padding type in future.
  string padding_str;
  TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "padding", &padding_str));
  if (padding_str == "SAME") return false;

  // Only fuse contraction with INT32 padding.
  // TODO(plugin): support INT64 padding in future.
  if (!HasDataType(pad_node_def, DT_INT32, "Tpaddings")) return false;

  // If contraction has been fused, only fuse it with Pad if, Conv is fused with
  // only Bias.
  if (node_def->op() == kFusedConv2D) {
    int num_args;
    TF_ABORT_IF_ERROR(GetNodeAttr(*node_def, "num_args", &num_args));
    if (num_args != 1) return false;
  }

  if (!HaveSameDataType(node_def, pad_node_def) ||
      HasControlFaninOrFanout(*pad_node_view) ||
      !HasAtMostOneFanoutAtPort0(*pad_node_view) ||
      IsInPreserveSet(ctx, pad_node_def))
    return false;

  const PadWithContraction pattern{pad_node_view->node_index(), node_index};

  // We successfully found a Pad + (_Fused)Conv2D/DepthwiseConv2dNative pattern.
  *matched = pattern;

  return true;
}

inline bool VerifyConstants(RemapperContext* ctx,
                            std::map<string, int>* nodes_map,
                            std::map<string, float>* values_map) {
  using utils::MutableNodeView;
  for (auto it = values_map->begin(); it != values_map->end(); ++it) {
    int node_idx = nodes_map->at(it->first);
    MutableNodeView* node_view = ctx->graph_view.GetNode(node_idx);
    NodeDef* node_def = node_view->node();
    Tensor const_tensor;

    // Check if node is Const or Cast.
    if (node_def != nullptr &&
        (node_def->op() == "Cast" || node_def->op() == "Const")) {
      // If node is a Cast, look for Const in fan-ins.
      if (node_def->op() == "Cast") {
        const auto& regular_fanin_0 = node_view->GetRegularFanin(0);
        const auto* regular_node_view = regular_fanin_0.node_view();
        node_def = regular_node_view->node();
      }
      // Verify if the node is a constant.
      if (node_def->op() == "Const") {
        TF_CHECK_OK(GetTensorFromConstant(node_def, &const_tensor));
        if (const_tensor.NumElements() == 1) {
          DataType dtype = const_tensor.dtype();
          if (!(dtype == DT_FLOAT || dtype == DT_BFLOAT16)) return false;
          auto const_value = (dtype == DT_FLOAT)
                                 ? const_tensor.flat<float>()(0)
                                 : const_tensor.flat<Eigen::bfloat16>()(0);
          // To compare float.
          if (std::abs(const_value - it->second) > 1e-2f) {
            return false;
          }
        } else {
          return false;
        }
      } else {
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
}

bool IsMatchedMatMulBiasAddAndGeluExact(
    RemapperContext& ctx, int node_index,
    std::map<string, int>* matched_nodes_map = nullptr,
    std::set<int>* remove_node_indices = nullptr, bool* expand_dims = nullptr) {
  auto* node_view = ctx.graph_view.GetNode(node_index);
  using utils::MatchingDirection;
  using utils::NodeStatus;
  int found_pattern_index = 0;  // Default = 0 means no pattern found.
  std::vector<utils::OpTypePattern> gelu_exact_patterns;
  // clang-format off
  // Pattern 1:
  //    Const: 1/sqrt(2)        Const: 1    Const: 1/2
  //                  \               \         \
  //  * --> BiasAdd --> Mul --> Erf --> AddV2 --> Mul --> Mul
  //        /       \____________________________________/
  //  MatMul
  gelu_exact_patterns.push_back(
    {"Mul", "output", NodeStatus::kReplace,
      {
        {"Mul", "erf_plus_one_times_one_half", NodeStatus::kRemove,
          {
            {"Add|AddV2", "erf_plus_one", NodeStatus::kRemove,
              {
                {"Erf", "erf", NodeStatus::kRemove,
                  {
                    {"Mul", "bias_add_x_sqrt_one_half",
                     NodeStatus::kRemove,
                      {
                        {"BiasAdd", "bias_add", NodeStatus::kRemove},
                        {"Cast|Const", "sqrt_one_half", NodeStatus::kRemain}
                      }
                    }  // Mul: "bias_add_x_sqrt_one_half"
                  }
                },  // Erf: "erf"
                {"Cast|Const", "one", NodeStatus::kRemain}
              }  // Add|AddV2: "erf_plus_one"
            },
            {"Cast|Const", "one_half", NodeStatus::kRemain}
          }
        },  // Mul: "erf_plus_one_times_one_half"
        {"BiasAdd", "bias_add", NodeStatus::kRemove,
          {
            {"MatMul", "matmul", NodeStatus::kRemove},
            {"*", "bias", NodeStatus::kRemain}
          }
        }  // BiasAdd: "bias_add"
      }  // Mul: "output"
    });

  // Pattern 2:
  //  Cast|Const: 1/sqrt(2)    Cast|Const: 1
  //                  \               \
  //  * --> BiasAdd --> Mul --> Erf --> Add|AddV2 --> Mul
  //      /         \                                 /
  // MatMul           ----------------------------> Mul
  //                                                /
  //                                  Cast|Const: 1/2
  gelu_exact_patterns.push_back(
    {"Mul", "output", NodeStatus::kReplace,
      {
        {"Add|AddV2", "erf_plus_one", NodeStatus::kRemove,
          {
            {"Erf", "erf", NodeStatus::kRemove,
              {
                {"Mul", "bias_add_x_sqrt_one_half", NodeStatus::kRemove,
                  {
                    {"BiasAdd", "bias_add", NodeStatus::kRemove},
                    {"Cast|Const", "sqrt_one_half", NodeStatus::kRemain}
                  }
                }  // Mul: "bias_add_x_sqrt_one_half"
              }
            },  // Erf: "erf"
            {"Cast|Const", "one", NodeStatus::kRemain}
          }
        },  // Add|AddV2: "erf_plus_one"
        {"Mul", "erf_plus_one_times_one_half", NodeStatus::kRemove,
          {
            {"BiasAdd", "bias_add", NodeStatus::kRemove,
              {
                {"MatMul", "matmul", NodeStatus::kRemove},
                {"*", "bias", NodeStatus::kRemain}
              }
            },  // BiasAdd: "bias_add"
            {"Cast|Const", "one_half", NodeStatus::kRemain}
          }
        }  // Mul: "erf_plus_one_times_one_half"
      }
    });  // Mul: "output"

  // Pattern 3:
  //    Const: expand_dims  1/sqrt(2)        Const: 1    Const: 1/2
  //                 \           \               \         \
  //  _FusedMatMul --> Reshape --> Mul --> Erf --> AddV2 --> Mul --> Mul
  //                           \____________________________________/
  gelu_exact_patterns.push_back(
    {"Mul", "output", NodeStatus::kReplace,
      {
        {"Mul", "erf_plus_one_times_one_half", NodeStatus::kRemove,
          {
            {"Add|AddV2", "erf_plus_one", NodeStatus::kRemove,
              {
                {"Erf", "erf", NodeStatus::kRemove,
                  {
                    {"Mul", "bias_add_x_sqrt_one_half", NodeStatus::kRemove,
                      {
                        {"Reshape", "reshape", NodeStatus::kRemove},
                        {"Cast|Const", "sqrt_one_half", NodeStatus::kRemain}
                      }
                    }  // Mul: "bias_add_x_sqrt_one_half"
                  }
                },  // Erf: "erf"
                {"Cast|Const", "one", NodeStatus::kRemain}
              }  // Add|AddV2: "erf_plus_one"
            },
            {"Cast|Const", "one_half", NodeStatus::kRemain}
          }
        },  // Mul: "erf_plus_one_times_one_half"
        {"Reshape", "reshape", NodeStatus::kRemove,
          {
            {"_FusedMatMul", "matmul", NodeStatus::kRemove},
            {"Cast|Const", "expand_dims", NodeStatus::kRemain}
          }
        }
      }
    });  // Mul: "output"

  // Pattern 4: Erfc
  //                                     Const: 1/sqrt(2) Const: 1/2
  //                                      \                \
  //  _FusedMatMul --> Reshape --> Neg --> Mul --> Erfc --> Mul --> Mul
  //                            \____________________________________/
  gelu_exact_patterns.push_back(
    {"Mul", "output", NodeStatus::kReplace,
      {
        {"Mul", "one_half_x_erfc", NodeStatus::kRemove,
          {
            {"Const|Cast", "one_half", NodeStatus::kRemain},
            {"Erfc", "erfc", NodeStatus::kRemove,
              {
                {"Mul", "neg_bias_add_x_sqrt_one_half", NodeStatus::kRemove,
                  {
                    {"Const|Cast", "sqrt_one_half", NodeStatus::kRemain},
                    {"Neg", "neg", NodeStatus::kRemove,
                      {{"Reshape", "reshape", NodeStatus::kRemove}}
                    },  // Neg: "neg"
                  }
                }  // Mul: "neg_bias_add_x_sqrt_one_half"
              }  // Erfc: "erfc"
            }
          }  // Mul: "one_half_x_erfc"
        },
        {"Reshape", "reshape", NodeStatus::kRemove,
          {
            {"_FusedMatMul", "matmul", NodeStatus::kRemove},
            {"Const|Cast", "expand_dims", NodeStatus::kRemain}
          }
        }
      }
    });  // Mul: "output"

  // clang-format on
  utils::SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(
      &(ctx.graph_view));
  // Find GeluExact
  std::map<string, int> dummy_matched_nodes_map;
  std::set<int> dummy_remove_node_indices;
  if (!matched_nodes_map) matched_nodes_map = &dummy_matched_nodes_map;
  if (!remove_node_indices) remove_node_indices = &dummy_remove_node_indices;
  bool found_gelu_exact = false;

  for (size_t pattern = 0; pattern < gelu_exact_patterns.size(); pattern++) {
    matched_nodes_map->clear();
    remove_node_indices->clear();
    if (graph_matcher.GetMatchedNodes(gelu_exact_patterns[pattern],
                                      ctx.nodes_to_preserve, node_view,
                                      matched_nodes_map, remove_node_indices)) {
      found_gelu_exact = true;
      found_pattern_index = pattern + 1;
      break;
    }
  }
  if (found_pattern_index == 3 || found_pattern_index == 4) {
    *expand_dims = true;
  }
  return found_gelu_exact;
}

// Gelu in python api generates a number of nodes in the graph. Depending on the
// parmeter `approximate={True/False}` different types of ops are generated. We
// distinguish them as `GeluExact` that uses Erf and `GeluApproximate` that
// uses Tanh.
bool FindMatMulBiasAddAndGelu(RemapperContext* ctx, int node_index,
                              std::map<string, int>* matched_nodes_map,
                              std::set<int>* remove_node_indices,
                              bool* is_gelu_approximate, bool* expand_dims) {
  using utils::MatchingDirection;
  using utils::NodeStatus;

  bool found_gelu_exact = false;
  bool found_gelu_approximate = false;
  int found_pattern_index = 0;
  std::vector<utils::OpTypePattern> gelu_approximate_patterns;

  // Find GeluExact
  matched_nodes_map->clear();
  remove_node_indices->clear();
  found_gelu_exact = IsMatchedMatMulBiasAddAndGeluExact(
      *ctx, node_index, matched_nodes_map, remove_node_indices, expand_dims);
  utils::SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(
      &(ctx->graph_view));
  // clang-format off
  // SubGraph gpu Pattern
  //            Const: 3  Empirical Const
  //                \      \
  //  FusedMatMul --> Pow --> Mul
  utils::OpTypePattern subgraph_gpu =
    {"Mul", "mul", NodeStatus::kRemove,
      {
        {"Pow", "pow", NodeStatus::kRemove,
          {
            {"_FusedMatMul", "matmul", NodeStatus::kRemove},
            {"Const", "three", NodeStatus::kRemain}
          }
        },
        {"Const", "empirical_const", NodeStatus::kRemain}
      }
    };

  // SubGraph cpu Pattern
  //            Empirical Const
  //                   |
  //             ------Mul------
  //            /               \
  //  FusedMatMul               Mul
  //            \               /
  //             -----Square----
  utils::OpTypePattern subgraph_cpu =
    {"Mul", "mul", NodeStatus::kRemove,
      {
        {"Mul", "empirical_const_times_matmul", NodeStatus::kRemove,
          {
            {"Const", "empirical_const", NodeStatus::kRemain},
            {"_FusedMatMul", "matmul", NodeStatus::kRemove}
          }
        },
        {"Square", "square", NodeStatus::kRemove,
          {
            {"_FusedMatMul", "matmul", NodeStatus::kRemove}
          }
        }
      }
    };
  // clang-format on

  utils::MutableNodeView* node_view = ctx->graph_view.GetNode(node_index);
  const NodeDef* node_def = node_view->node();
  bool root_on_gpu = NodeIsOnGpu(node_def);
  utils::OpTypePattern* subgraph_pattern =
      root_on_gpu ? &subgraph_gpu : &subgraph_cpu;

  // clang-format off
  // Pattern 1:
  //             Const: 1/sqrt(2)       Const: 1    Const: 1/2
  //            SubGraph     \                \        \
  //                \         \                \        \
  //  FusedMatMul --> AddV2 --> Mul --> Tanh --> AddV2 --> Mul --> Mul
  //            \________________________________________________/
  gelu_approximate_patterns.push_back(
    {"Mul", "output", NodeStatus::kReplace,
      {
        {"Mul", "tanh_plus_one_times_one_half", NodeStatus::kRemove,
          {
            {"Add|AddV2", "tanh_plus_one", NodeStatus::kRemove,
              {
                {"Tanh", "tanh", NodeStatus::kRemove,
                  {
                    {"Mul", "matmul_plus_mul_times_square_root_two_over_pi", NodeStatus::kRemove,
                      {
                        {"Add|AddV2", "matmul_plus_mul", NodeStatus::kRemove,
                          {
                            {"_FusedMatMul", "matmul", NodeStatus::kRemove},
                            *subgraph_pattern
                          }
                        },  // Add|AddV2: matmul_plus_mul
                        {"Const", "square_root_two_over_pi", NodeStatus::kRemain}
                      }
                    }  // Mul: matmul_plus_mul_times_square_root_two_over_pi
                  }
                },  // Tanh: tanh
                {"Const", "one", NodeStatus::kRemain}
              }
            },  // Add|AddV2: tanh_plus_one
            {"Const", "one_half", NodeStatus::kRemain}
          }
        },  // Mul: tanh_plus_one_times_one_half
        {"_FusedMatMul", "matmul", NodeStatus::kRemove}
      }
    });  // Mul: output

  // Pattern 2:
  //    Const:                      Emperical     1/sqrt(2)             1
  //                                   \                 \               \
  //  FusedMatMul --> Square --> Mul --> Mul --> AddV2 --> Mul --> Tanh --> AddV2  --> Mul
  //           \\\_______________/              /                                    /
  //            \\_____________________________/                                    /
  //             \______________________________________________________________ Mul
  //                                                                             /
  //                                                                         Const: 1/2
  gelu_approximate_patterns.push_back(
  {"Mul", "output", NodeStatus::kReplace,
    {
      {"Mul", "tanh_plus_one_times_one_half", NodeStatus::kRemove,
        {
          {"Const", "one_half", NodeStatus::kRemain},
          {"_FusedMatMul", "matmul", NodeStatus::kRemove}
        }
      },  // Mul: tanh_plus_one_times_one_half
      {"Add|AddV2", "tanh_plus_one", NodeStatus::kRemove,
        {
          {"Tanh", "tanh", NodeStatus::kRemove,
            {
              {"Mul", "matmul_plus_mul_times_square_root_two_over_pi", NodeStatus::kRemove,
                {
                  {"Add|AddV2", "matmul_plus_mul", NodeStatus::kRemove,
                    {
                      {"_FusedMatMul", "matmul", NodeStatus::kRemove},
                      {"Mul", "mul", NodeStatus::kRemove,
                        {
                          {"Const", "empirical_const", NodeStatus::kRemain},
                          {"Mul", "empirical_const_times_matmul", NodeStatus::kRemove,
                            {
                              {"_FusedMatMul", "matmul", NodeStatus::kRemove},
                              {"Square", "square", NodeStatus::kRemove,
                                {
                                  {"_FusedMatMul", "matmul", NodeStatus::kRemove}
                                }
                              }  // Square: square
                            }
                          }  // Mul: empirical_const_times_matmul
                        }
                      }  // Mul: mul
                    }
                  },  // Add|AddV2: matmul_plus_mul
                  {"Const", "square_root_two_over_pi", NodeStatus::kRemain}
                }
              }  // Mul: matmul_plus_mul_times_square_root_two_over_pi
            }
          },  // Tanh: tanh
          {"Const", "one", NodeStatus::kRemain}
        }
      }  // Add|AddV2: tanh_plus_one
    }
  });  // Mul: output

  // Pattern 3:
  //    Const:   ExpandDims                   Emperical        1/sqrt(2)       Const: 1    Const: 1/2
  //                \                              \                 \                \         \
  //  FusedMatMul --> Reshape --> Square --> Mul --> Mul --> AddV2 --> Mul --> Tanh --> AddV2 --> Mul --> Mul
  //                      \\\_______________/              /                                            /
  //                       \\_____________________________/                                            /
  //                        \_________________________________________________________________________/
  gelu_approximate_patterns.push_back(
  {"Mul", "output", NodeStatus::kReplace,
    {
      {"Mul", "tanh_plus_one_times_one_half", NodeStatus::kRemove,
        {
          {"Add|AddV2", "tanh_plus_one", NodeStatus::kRemove,
            {
              {"Tanh", "tanh", NodeStatus::kRemove,
                {
                  {"Mul", "matmul_plus_mul_times_square_root_two_over_pi", NodeStatus::kRemove,
                    {
                      {"Add|AddV2", "matmul_plus_mul", NodeStatus::kRemove,
                        {
                          {"Reshape", "reshape", NodeStatus::kRemove},
                          {"Mul", "mul", NodeStatus::kRemove,
                            {
                              {"Mul", "empirical_const_times_matmul", NodeStatus::kRemove,
                                {
                                  {"Reshape", "reshape", NodeStatus::kRemove,
                                    {
                                      {"_FusedMatMul", "matmul", NodeStatus::kRemove},
                                      {"Cast|Const", "expand_dims", NodeStatus::kRemain}
                                    }
                                  },  // Reshape: reshape
                                  {"Square", "square", NodeStatus::kRemove,
                                    {
                                      {"Reshape", "reshape", NodeStatus::kRemove}
                                    }
                                  }  // Square: square
                                }
                              },  // Mul: empirical_const_times_matmul
                              {"Cast|Const", "empirical_const", NodeStatus::kRemain}
                            }
                          }  // Mul: mul
                        }
                      },  // Add|AddV2: matmul_plus_mul
                      {"Cast|Const", "square_root_two_over_pi", NodeStatus::kRemain}
                    }
                  }  // Mul: matmul_plus_mul_times_square_root_two_over_pi
                }
              },  // Tanh: tanh
              {"Cast|Const", "one", NodeStatus::kRemain}
            }
          },  // Add|AddV2: tanh_plus_one
          {"Cast|Const", "one_half", NodeStatus::kRemain}
        }
      },  // Mul: tanh_plus_one_times_one_half
      {"Reshape", "reshape", NodeStatus::kRemove}
    }
  });  // Mul: output

  // Pattern BF16:
  //    Const:                    Emperical                1/sqrt(2)                      1
  //                                   \                         \                         \
  //  FusedMatMul --> Cast --> Square --> Mul --> Mul --> AddV2 --> Mul --> Cast --> Tanh --> AddV2  --> Mul
  //            \      \\\______________________/       /                                             /
  //             \      \\_____________________________/                                             /
  //              \______________________________________________________________________________ Mul
  //                                                                                              /
  //                                                                                          Const: 1/2
  gelu_approximate_patterns.push_back(
  {"Mul", "output", NodeStatus::kReplace,
    {
      {"Mul", "tanh_plus_one_times_one_half", NodeStatus::kRemove,
        {
          {"Const", "one_half", NodeStatus::kRemain},
          {"_FusedMatMul", "matmul", NodeStatus::kRemove}
        }
      },  // Mul: tanh_plus_one_times_one_half
      {"AddV2", "tanh_plus_one", NodeStatus::kRemove,
        {
          {"Tanh", "tanh", NodeStatus::kRemove,
            {
              {"Cast", "cast1", NodeStatus::kRemove,
                {
                  {"Mul", "matmul_plus_mul_times_square_root_two_over_pi", NodeStatus::kRemove,
                    {
                      {"AddV2", "matmul_plus_mul", NodeStatus::kRemove,
                        {
                          {"Cast", "cast", NodeStatus::kRemove,
                            {
                              {"_FusedMatMul", "matmul", NodeStatus::kRemove}
                            }
                          },  // Cast: cast
                          {"Mul", "mul", NodeStatus::kRemove,
                            {
                              {"Cast", "cast", NodeStatus::kRemove,
                                {
                                  {"_FusedMatMul", "matmul", NodeStatus::kRemove}
                                }
                              },  // Cast: cast
                              {"Mul", "empirical_const_times_matmul", NodeStatus::kRemove,
                                {
                                  {"Const", "empirical_const", NodeStatus::kRemain},
                                  {"Square", "square", NodeStatus::kRemove,
                                    {
                                      {"Cast", "cast", NodeStatus::kRemove,
                                        {
                                          {"_FusedMatMul", "matmul", NodeStatus::kRemove}
                                        }
                                      }  // Cast: cast
                                    }
                                  }  // Square: square
                                }
                              }  // Mul: empirical_const_times_matmul
                            }
                          }  // Mul: mul
                        }
                      },  // AddV2: matmul_plus_mul
                      {"Const", "square_root_two_over_pi", NodeStatus::kRemain}
                    }
                  }  // Mul: matmul_plus_mul_times_square_root_two_over_pi
                }
              }  // Cast: cast1
            }
          },  // Tanh: tanh
          {"Const", "one", NodeStatus::kRemain}
        }
      }  // AddV2: tanh_plus_one
    }
  });  // Mul: output

  // clang-format on
  // Find GeluApproximate
  if (!found_gelu_exact) {
    for (size_t pattern = 0; pattern < gelu_approximate_patterns.size();
         pattern++) {
      matched_nodes_map->clear();
      remove_node_indices->clear();
      if (graph_matcher.GetMatchedNodes(
              gelu_approximate_patterns[pattern], ctx->nodes_to_preserve,
              node_view, matched_nodes_map, remove_node_indices)) {
        found_gelu_approximate = true;
        found_pattern_index = pattern + 1;
        break;
      }
    }
    if (found_pattern_index == 3) *expand_dims = true;
  }

  *is_gelu_approximate = found_gelu_approximate ? true : false;
  // Pattern matcher does subgraph matching based on op types only. The matcher
  // also does a sanity check on nodes tagged as `kRemove`, i.e., they do not
  // have any consumer outside the matched nodes. In order to replace the
  // subgraph, we need additional checks, for example, if the key ops have been
  // placed on CPU or GPU, desired data type, const has desired value etc. For
  // the following fusion: MatMul + BiasAdd + Gelu (disintegrated into smaller
  // ops), we check if (i) MatMul op is CpuCompatible or GpuComptible, (ii)
  // const nodes have desired values.
  if (found_gelu_exact) {
    // Check if the MatMul to be fused is device compatible.
    NodeDef* matmul_node =
        ctx->graph_view.GetNode(matched_nodes_map->at("matmul"))->node();

    if (!HasDataType(node_def, DT_FLOAT) && !HasDataType(node_def, DT_BFLOAT16))
      return false;
    // Currently, the fusion is not supported on CPU for transpose_a in the
    // MatMul op.
    bool cpu_ok = matmul_node->attr().contains("transpose_a") &&
                  !matmul_node->attr().at("transpose_a").b();
    if (!cpu_ok) return false;

    // Check if the matched constants have desired values.
    std::map<string, float> values_map = {{"sqrt_one_half", 0.707106},
                                          {"one_half", 0.5}};

    // Gelu exact pattern with TF version 2.19 does not have the constant one.
    // Check is added to see if the matched pattern has this constant, if True
    // it is added. For TF version 2.18 it is added.
    if (matched_nodes_map->find("one") != matched_nodes_map->end()) {
      values_map["one"] = 1.0;
    }
    if (!VerifyConstants(ctx, matched_nodes_map, &values_map)) return false;
  } else if (*is_gelu_approximate) {
    NodeDef* matmul_node =
        ctx->graph_view.GetNode(matched_nodes_map->at("matmul"))->node();

    // Currently, the fusion is not supported on CPU for transpose_a in the
    // MatMul op.
    if (NodeIsOnCpu(matmul_node) &&
        matmul_node->attr().contains("transpose_a") &&
        matmul_node->attr().at("transpose_a").b()) {
      return false;
    }

    // Check if _FusedMatMul contains only BiasAdd
    auto fused_ops = matmul_node->attr().at("fused_ops").list().s();
    if (fused_ops.size() == 1) {
      if (fused_ops.at(0) != "BiasAdd") return false;
    } else {
      return false;
    }
    // Check if the matched constants have desired values.
    std::map<string, float> values_map = {{"square_root_two_over_pi", 0.797884},
                                          {"one", 1.0},
                                          {"one_half", 0.5},
                                          {"empirical_const", 0.044715}};

    if (!VerifyConstants(ctx, matched_nodes_map, &values_map)) return false;
  } else {
    return false;
  }

  return (found_gelu_exact || *is_gelu_approximate);
}

bool FindFusedBatchMatMul(RemapperContext* ctx, int node_index,
                          std::map<string, int>* matched_nodes_map,
                          std::set<int>* remove_node_indices,
                          std::vector<string>* input_node_names) {
  const auto* node_view = ctx->graph_view.GetNode(node_index);
  const auto* node_def = node_view->node();
  if (node_def == nullptr) return false;
  // The fusion is only supported for FP32.
  if (!HasDataType(node_def, DT_FLOAT)) return false;
  using utils::MatchingDirection;
  using utils::NodeStatus;
  int found_pattern_index = 0;  // Default = 0 means no pattern found.
  std::vector<utils::OpTypePattern> fusion_patterns;
  // clang-format off
  fusion_patterns.push_back(
    {"Add|AddV2", "output", NodeStatus::kReplace,
      {
        {"Mul", "mul", NodeStatus::kRemove,
          {
            {"BatchMatMulV2", "batch_matmul", NodeStatus::kRemove},
            {"*", "multiplicand", NodeStatus::kRemain}
          }
        },
        {"*", "addend", NodeStatus::kRemain}
      }
    });

  fusion_patterns.push_back(
    {"Add|AddV2", "output", NodeStatus::kReplace,
      {
        {"BatchMatMulV2", "batch_matmul", NodeStatus::kRemove,
          {
            {"Mul", "mul", NodeStatus::kRemove,
              {
                {"*", "mul_input0", NodeStatus::kRemain},
                {"Const|Cast", "multiplicand", NodeStatus::kRemain}
              }
            },
            {"*", "bmm_input1", NodeStatus::kRemain}
          }
        },
        {"*", "addend", NodeStatus::kRemain}
      }
    });

  fusion_patterns.push_back(
    {"Add|AddV2", "output", NodeStatus::kReplace,
      {
        {"*", "addend", NodeStatus::kRemain},
        {"Mul", "mul", NodeStatus::kRemove,
          {
            {"BatchMatMulV2", "batch_matmul", NodeStatus::kRemove},
            {"*", "multiplicand", NodeStatus::kRemain}
          }
        }
      }
    });
  // clang-format on

  utils::SubGraphMatcher<MatchingDirection::kFollowInputs> graph_matcher(
      &(ctx->graph_view));

  bool found_op_type_match = false;

  for (size_t pattern_iterator = 0; pattern_iterator < fusion_patterns.size();
       pattern_iterator++) {
    matched_nodes_map->clear();
    remove_node_indices->clear();
    found_op_type_match = graph_matcher.GetMatchedNodes(
        fusion_patterns[pattern_iterator], ctx->nodes_to_preserve,
        ctx->graph_view.GetNode(node_index), matched_nodes_map,
        remove_node_indices);
    if (found_op_type_match) {
      found_pattern_index = pattern_iterator + 1;
      break;
    }
  }

  // ZenDNN is not optimized for all shapes with regard to binary-post ops
  // fusion. Allow limited cases only for now that are optimized, (i)
  // multiplicand is scalar, (ii) BatchMatmulV2 output is 4D tensor, and (iii)
  // addend is 4D tensor with second dim_size = 1.
  if (!found_op_type_match) return false;
  if (!ctx->inferred_graph_properties) {
    Status s = ctx->graph_properties.InferStatically(
        /*assume_valid_feeds=*/true,
        /*aggressive_shape_inference=*/false,
        /*include_input_tensor_values=*/false,
        /*include_output_tensor_values=*/true);
    if (!s.ok()) return false;
    ctx->inferred_graph_properties = true;
  }
  std::vector<OpInfo_TensorProperties> multiplicand_props;
  NodeDef* multiplicand_node_def =
      ctx->graph_view.GetNode(matched_nodes_map->at("multiplicand"))->node();
  TF_ABORT_IF_ERROR(ctx->graph_properties.GetOutputProperties(
      multiplicand_node_def->name(), &multiplicand_props));
  if (NumCoefficients(multiplicand_props[0].shape()) != 1) return false;

  NodeDef* batch_matmul_node_def =
      ctx->graph_view.GetNode(matched_nodes_map->at("batch_matmul"))->node();
  DCHECK(IsAnyBatchMatMul(*batch_matmul_node_def));  // Expected BatchMatMul op.
  if (!NodeIsOnCpu(batch_matmul_node_def)) return false;

  std::vector<OpInfo_TensorProperties> batch_matmul_props;
  TF_ABORT_IF_ERROR(ctx->graph_properties.GetOutputProperties(
      batch_matmul_node_def->name(), &batch_matmul_props));
  if (Rank(batch_matmul_props[0].shape()) != 4) return false;

  std::vector<OpInfo_TensorProperties> addend_props;
  NodeDef* addend_node_def =
      ctx->graph_view.GetNode(matched_nodes_map->at("addend"))->node();
  TF_ABORT_IF_ERROR(ctx->graph_properties.GetOutputProperties(
      addend_node_def->name(), &addend_props));
  auto addend_shape = addend_props[0].shape();
  if (!(Rank(addend_shape) == 4 && addend_shape.dim(1).size() == 1)) {
    return false;
  }
  input_node_names->clear();
  input_node_names->resize(4);
  if (found_pattern_index == 1 || found_pattern_index == 3) {
    input_node_names->at(0) = batch_matmul_node_def->input(0);
  } else if (found_pattern_index == 2) {
    auto* mul_input0_node_def =
        ctx->graph_view.GetNode(matched_nodes_map->at("mul_input0"))->node();
    input_node_names->at(0) = mul_input0_node_def->name();
  }
  input_node_names->at(1) = batch_matmul_node_def->input(1);
  input_node_names->at(2) = multiplicand_node_def->name();
  input_node_names->at(3) = addend_node_def->name();
  return found_op_type_match;
}

// Fuse BatchMatMul and Mul into FusedBatchMatmul if the other input of
// Mul is a scalar. For example, we can optimize:
/*
              Mul
             /  \
    BatchMatMul scale*  ->       FusedBatchMatmul
       /   \                     /      |       \
   input1  input2             input1  input2   scale
*/
// *) scale must be a scalar.

bool FindContractionWithMul(const RemapperContext& ctx, int node_index,
                            ContractionWithMul* matched) {
  const auto* node_view = ctx.graph_view.GetNode(node_index);
  if (HasControlFaninOrFanout(*node_view)) return false;

  const auto* node_def = node_view->node();
  if (!IsAnyMul(*node_def)) return false;
  // Mul has two inputs, one input should be scalar.
  int scalar_input_index = GetMulScalarInputIndex(ctx, *node_def);
  if (scalar_input_index == -1) return false;

  auto* const_node_view =
      node_view->GetRegularFanin(scalar_input_index).node_view();
  auto* contraction_node_view =
      node_view->GetRegularFanin(1 - scalar_input_index).node_view();

  auto* contraction_node_def = contraction_node_view->node();
  if (!IsAnyBatchMatMul(*contraction_node_def)) return false;
  auto* const_node_def = const_node_view->node();
  if (!(IsAnyConst(*const_node_def) || IsCast(*const_node_def))) return false;

  // The fusion is only supported for float and bfloat16 on CPU.
  if (!HasDataType(node_def, DT_FLOAT) && !HasDataType(node_def, DT_BFLOAT16))
    return false;

  if (!HaveSameDataType(node_def, contraction_node_def) ||
      HasControlFaninOrFanout(*contraction_node_view) ||
      !HasAtMostOneFanoutAtPort0(*contraction_node_view) ||
      IsInPreserveSet(ctx, contraction_node_def))
    return false;

  const ContractionWithMul pattern{contraction_node_view->node_index(),
                                   node_index, const_node_view->node_index()};

  *matched = pattern;

  return true;
}

void CopyBatchMatMulAttributes(const NodeDef& batchmatmul,
                               NodeDef* fused_batch_matmul) {
  DCHECK(IsAnyBatchMatMul(batchmatmul)) << "Input node must be a BatchMatMul";

  auto* attr = fused_batch_matmul->mutable_attr();
  auto& src_attr = batchmatmul.attr();

  (*attr)["T"] = src_attr.at("T");
  (*attr)["adj_x"] = src_attr.at("adj_x");
  (*attr)["adj_y"] = src_attr.at("adj_y");

  // TODO(plugin): Validate if "input_shapes" is necessary for ZenDNN ops.
  auto input_shapes = src_attr.find("_input_shapes");
  if (input_shapes != src_attr.end()) {
    (*attr)["_input_shapes"] = input_shapes->second;
  }
}

void CopyConv2DAttributes(const NodeDef& conv2d, NodeDef* fused_conv2d,
                          const NodeDef* activation = nullptr) {
  DCHECK(IsConv2D(conv2d)) << "Input node must be a Conv2D";
  auto* attr = fused_conv2d->mutable_attr();
  auto& src_attr = conv2d.attr();

  (*attr)["T"] = src_attr.at("T");
  int num_args = fused_conv2d->input_size() - 2;
  for (int i = 0; i < num_args; ++i) {
    (*attr)["TArgs"].mutable_list()->add_type(src_attr.at("T").type());
  }
  (*attr)["num_host_args"].set_i(0);
  (*attr)["strides"] = src_attr.at("strides");
  (*attr)["padding"] = src_attr.at("padding");
  (*attr)["explicit_paddings"] = src_attr.at("explicit_paddings");
  (*attr)["dilations"] = src_attr.at("dilations");
  (*attr)["data_format"] = src_attr.at("data_format");
  (*attr)["use_cudnn_on_gpu"] = src_attr.at("use_cudnn_on_gpu");
  // Copy LeakyRelu's attr alpha to FusedConv2D's attr leakyrelu_alpha.
  if (activation != nullptr && IsLeakyRelu(*activation)) {
    auto& activation_attr = activation->attr();
    (*attr)["leakyrelu_alpha"] = activation_attr.at("alpha");
  }
}

void CopyDepthwiseConv2dNativeAttributes(const NodeDef& dw_conv2d,
                                         NodeDef* fused_dw_conv2d,
                                         const NodeDef* activation = nullptr) {
  DCHECK(IsDepthwiseConv2dNative(dw_conv2d))
      << "Input node must be a DepthwiseConv2dNative";

  auto* attr = fused_dw_conv2d->mutable_attr();
  auto& src_attr = dw_conv2d.attr();

  (*attr)["T"] = src_attr.at("T");
  (*attr)["strides"] = src_attr.at("strides");
  (*attr)["padding"] = src_attr.at("padding");
  (*attr)["dilations"] = src_attr.at("dilations");
  (*attr)["data_format"] = src_attr.at("data_format");
  if (HasNodeAttr(dw_conv2d, "explicit_paddings")) {
    (*attr)["explicit_paddings"] = src_attr.at("explicit_paddings");
  }
  // Copy LeakyRelu's attr alpha to FusedDepthwiseConv2d's attr leakyrelu_alpha.
  if (activation != nullptr && IsLeakyRelu(*activation)) {
    auto& activation_attr = activation->attr();
    (*attr)["leakyrelu_alpha"] = activation_attr.at("alpha");
  }
}

void CopyMatMulAttributes(const NodeDef& matmul, NodeDef* fused_matmul,
                          const NodeDef* activation = nullptr) {
  DCHECK(IsMatMul(matmul)) << "Input node must be a MatMul";

  auto* attr = fused_matmul->mutable_attr();
  auto& src_attr = matmul.attr();

  (*attr)["T"] = src_attr.at("T");
  (*attr)["transpose_a"] = src_attr.at("transpose_a");
  (*attr)["transpose_b"] = src_attr.at("transpose_b");
  // Copy LeakyRelu's attr alpha to _FusedMatMul's attr leakyrelu_alpha.
  // TODO(plugin) : Enable this when supporting LeakyRelu as fused activation.
  // if (activation != nullptr && IsLeakyRelu(*activation)) {
  //   auto& activation_attr = activation->attr();
  //   (*attr)["leakyrelu_alpha"] = activation_attr.at("alpha");
  // }
}

void CopyFusedBatchNormAttributes(const NodeDef& fused_batch_norm,
                                  NodeDef* fused_batch_norm_ex) {
  DCHECK(IsFusedBatchNorm(fused_batch_norm))
      << "Input node must be a FusedBatchNorm";

  CopyAllAttrs(fused_batch_norm, fused_batch_norm_ex);

  // FusedBatchNorm doesn't have an extra type parameter.
  if ((fused_batch_norm.op() == "FusedBatchNorm") ||
      (fused_batch_norm.op() == "FusedBatchNormGrad")) {
    AddNodeAttr("U", DT_FLOAT, fused_batch_norm_ex);
  }
}

void CopyReshapeAttributes(const NodeDef& reshape, NodeDef* node) {
  DCHECK(IsReshape(reshape)) << "Input node must be a Reshape";

  auto* attr = node->mutable_attr();
  auto& src_attr = reshape.attr();

  (*attr)["T"] = src_attr.at("T");
  (*attr)["Tshape"] = src_attr.at("Tshape");
}

Status AddKerasDenseLayerFwd(RemapperContext* ctx,
                             const KerasDenseLayerFwd& matched,
                             std::vector<bool>* invalidated_nodes,
                             std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& matmul = graph->node(matched.matmul);
  const NodeDef& reshape = graph->node(matched.reshape);
  const NodeDef& bias = graph->node(matched.bias);
  NodeDef new_shape;
  NodeDef fused_node;
  if (matched.activation != kMissingIndex) {
    const NodeDef& activation = graph->node(matched.activation);
    fused_node.set_op(kFusedMatMul);
    fused_node.set_name(bias.name());
    fused_node.set_device(matmul.device());
    fused_node.add_input(matmul.input(0));
    fused_node.add_input(matmul.input(1));
    fused_node.add_input(bias.input(1));
    CopyMatMulAttributes(matmul, &fused_node);
    SetFusedOpAttributesWithActivation(&fused_node, &activation, {"BiasAdd"});
    NodeDef new_reshape;
    new_reshape.set_op(kReshape);
    new_reshape.set_device(reshape.device());
    new_reshape.set_name(activation.name());
    new_reshape.add_input(bias.name());
    new_reshape.add_input(reshape.input(1));
    CopyReshapeAttributes(reshape, &new_reshape);

    utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
    Status status;

    mutation->AddNode(std::move(fused_node), &status);
    mutation->AddNode(std::move(new_reshape), &status);
    (*invalidated_nodes)[matched.activation] = true;
    (*invalidated_nodes)[matched.bias] = true;
    (*nodes_to_delete)[matched.reshape] = true;
    (*nodes_to_delete)[matched.matmul] = true;

    TF_ABORT_IF_ERROR(status);
    TF_ABORT_IF_ERROR(mutation->Apply());
    return OkStatus();
  } else {
    fused_node.set_op(kFusedMatMul);
    fused_node.set_name(reshape.name());
    fused_node.set_device(matmul.device());
    fused_node.add_input(matmul.input(0));
    fused_node.add_input(matmul.input(1));
    fused_node.add_input(bias.input(1));
    CopyMatMulAttributes(matmul, &fused_node);
    SetFusedOpAttributes(&fused_node, {"BiasAdd"});
    NodeDef new_reshape;
    new_reshape.set_op(kReshape);
    new_reshape.set_device(reshape.device());
    new_reshape.set_name(bias.name());
    new_reshape.add_input(reshape.name());
    new_reshape.add_input(reshape.input(1));
    CopyReshapeAttributes(reshape, &new_reshape);

    utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
    Status status;

    mutation->AddNode(std::move(fused_node), &status);
    mutation->AddNode(std::move(new_reshape), &status);
    (*invalidated_nodes)[matched.bias] = true;
    (*invalidated_nodes)[matched.reshape] = true;
    (*nodes_to_delete)[matched.matmul] = true;

    TF_ABORT_IF_ERROR(status);
    TF_ABORT_IF_ERROR(mutation->Apply());
    return OkStatus();
  }
}

// Contraction + BiasAdd.
Status AddFusedContractionNode(RemapperContext* ctx,
                               const ContractionWithBiasAdd& matched,
                               std::vector<bool>* invalidated_nodes,
                               std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);
  const NodeDef& bias_add = graph->node(matched.bias_add);
  zendnnInfo(ZENDNN_FWKLOG, "Fuse ", contraction.op(),
             " with BiasAdd: bias_add=", bias_add.name(),
             " contraction = ", contraction.name());

  NodeDef fused_node;
  fused_node.set_name(bias_add.name());
  fused_node.set_device(contraction.device());
  fused_node.add_input(contraction.input(0));               // 0: input
  fused_node.add_input(contraction.input(1));               // 1: filter
  fused_node.add_input(bias_add.input(matched.bias_port));  // 2: bias

  if (IsConv2D(contraction)) {
    fused_node.set_op(kFusedConv2D);
    CopyConv2DAttributes(contraction, &fused_node);
  } else if (IsDepthwiseConv2dNative(contraction)) {
    fused_node.set_op(kFusedDepthwiseConv2dNative);
    CopyDepthwiseConv2dNativeAttributes(contraction, &fused_node);
  } else if (IsMatMul(contraction)) {
    fused_node.set_op(kFusedMatMul);
    CopyMatMulAttributes(contraction, &fused_node);
    // TODO(plugin) : Explore if _ZenFusedBatchMatMul is a simple possibility.
  } else {
    CHECK(false);
  }

  SetFusedOpAttributes(&fused_node, {"BiasAdd"});

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_node), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.bias_add] = true;
  (*nodes_to_delete)[matched.contraction] = true;

  return OkStatus();
}

// Contraction + BiasAdd + Add.
Status AddFusedContractionNode(RemapperContext* ctx,
                               const ContractionWithBiasAddAndAdd& matched,
                               std::vector<bool>* invalidated_nodes,
                               std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);
  const NodeDef& bias_add = graph->node(matched.bias_add);
  const NodeDef& add = graph->node(matched.add);

  // ZenDNN only supports fusion for Conv and MatMul.
  DCHECK(IsConvOrMatMul(contraction));

  zendnnInfo(ZENDNN_FWKLOG, "Fuse ", contraction.op(), " with BiasAdd ",
             bias_add.op(), " and Add ", add.op(),
             " : bias_add=", bias_add.name(), " add=", add.name(),
             " contraction=", contraction.name());

  NodeDef fused_node;
  fused_node.set_name(add.name());
  fused_node.set_device(contraction.device());
  fused_node.add_input(contraction.input(0));  // 0: input(conv) / a (matmul)
  fused_node.add_input(contraction.input(1));  // 1: filter(conv) / b (matmul)
  fused_node.add_input(bias_add.input(matched.bias_port));  // 2: bias

  // Add OP has two inputs, one is conv+bias/matmul+bias pattern matched
  // previously, the other input to add is fused here.
  fused_node.add_input(add.input(1 - matched.port_id));

  if (IsConv2D(contraction)) {
    fused_node.set_op(kFusedConv2D);
    CopyConv2DAttributes(contraction, &fused_node);
  } else if (IsMatMul(contraction)) {
    fused_node.set_op(kFusedMatMul);
    CopyMatMulAttributes(contraction, &fused_node);
  } else {
    CHECK(false);
  }

  SetFusedOpAttributes(&fused_node, {"BiasAdd", "Add"}, 2);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_node), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.add] = true;
  (*nodes_to_delete)[matched.contraction] = true;
  (*nodes_to_delete)[matched.bias_add] = true;

  return OkStatus();
}

// Contraction + BiasAdd + Activation.
Status AddFusedContractionNode(
    RemapperContext* ctx, const ContractionWithBiasAddAndActivation& matched,
    std::vector<bool>* invalidated_nodes, std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);
  const NodeDef& bias_add = graph->node(matched.bias_add);
  const NodeDef& activation = graph->node(matched.activation);

  zendnnInfo(ZENDNN_FWKLOG, "Fuse ", contraction.op(), " with BiasAdd and ",
             activation.op(), ":", " activation=", activation.name(),
             " bias_add=", bias_add.name(),
             " contraction=", contraction.name());

  NodeDef fused_node;
  fused_node.set_name(activation.name());
  fused_node.set_device(contraction.device());
  fused_node.add_input(contraction.input(0));               // 0: input
  fused_node.add_input(contraction.input(1));               // 1: filter
  fused_node.add_input(bias_add.input(matched.bias_port));  // 2: bias

  if (IsConv2D(contraction)) {
    fused_node.set_op(kFusedConv2D);
    CopyConv2DAttributes(contraction, &fused_node);
  } else if (IsDepthwiseConv2dNative(contraction)) {
    fused_node.set_op(kFusedDepthwiseConv2dNative);
    CopyDepthwiseConv2dNativeAttributes(contraction, &fused_node);
  } else if (IsMatMul(contraction)) {
    fused_node.set_op(kFusedMatMul);
    CopyMatMulAttributes(contraction, &fused_node);
  } else {
    CHECK(false);
  }

  SetFusedOpAttributesWithActivation(&fused_node, &activation, {"BiasAdd"});

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_node), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*nodes_to_delete)[matched.contraction] = true;
  (*nodes_to_delete)[matched.bias_add] = true;
  (*invalidated_nodes)[matched.activation] = true;

  return OkStatus();
}

Status AddFusedMatMulSigmoidNode(RemapperContext* ctx,
                                 const ContractionWithActivation& matched,
                                 std::vector<bool>* invalidated_nodes,
                                 std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction =
      graph->node(matched.contraction);  // _FusedMatMul (MatMul + BiasAdd).
  const NodeDef& activation = graph->node(matched.activation);  // Sigmoid.

  // Log the fusion operation.
  zendnnInfo(ZENDNN_FWKLOG, "Fuse ", contraction.op(), " with ",
             activation.op(), ":", " activation=", activation.name(),
             " contraction=", contraction.name());

  // Create the new fused node.
  NodeDef fused_node;
  fused_node.set_name(activation.name());  // Name it after the Sigmoid node.
  fused_node.set_device(
      contraction.device());  // Use the device of the original _FusedMatMul.
  fused_node.add_input(
      contraction.input(0));  // Input tensor (e.g., from concat).
  fused_node.add_input(contraction.input(1));  // Filter (constant kernel).
  fused_node.add_input(contraction.input(2));  // Bias (constant bias).

  // Set the operation to _FusedMatMul.
  fused_node.set_op(kFusedMatMul);

  // Copy attributes from the original _FusedMatMul.
  CopyMatMulAttributes(contraction, &fused_node);

  // Set the fused operations to include both BiasAdd and Sigmoid.
  SetFusedOpAttributesWithActivation(&fused_node, &activation, {"BiasAdd"});

  // Add the new node to the graph.
  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_node), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  // Mark the original _FusedMatMul for deletion and the Sigmoid as invalidated.
  (*nodes_to_delete)[matched.contraction] = true;
  (*invalidated_nodes)[matched.activation] = true;

  return OkStatus();
}

// Contraction + Activation.
Status AddFusedContractionNode(RemapperContext* ctx,
                               const ContractionWithActivation& matched,
                               std::vector<bool>* invalidated_nodes,
                               std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);
  const NodeDef& activation = graph->node(matched.activation);

  zendnnInfo(ZENDNN_FWKLOG, "Fuse ", contraction.op(), " with ",
             activation.op(), ":", " activation=", activation.name(),
             " contraction=", contraction.name());

  NodeDef fused_node;
  fused_node.set_name(activation.name());
  fused_node.set_device(contraction.device());
  fused_node.add_input(contraction.input(0));  // 0: input
  fused_node.add_input(contraction.input(1));  // 1: filter

  if (IsMatMul(contraction)) {
    fused_node.set_op(kFusedMatMul);
    CopyMatMulAttributes(contraction, &fused_node);
  } else {
    CHECK(false);
  }

  SetFusedOpAttributesWithActivation(&fused_node, &activation, {}, 0);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_node), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*nodes_to_delete)[matched.contraction] = true;
  (*invalidated_nodes)[matched.activation] = true;

  return OkStatus();
}

// Contraction + BiasAdd + Add + Activation.
Status AddFusedContractionNode(
    RemapperContext* ctx, const ContractionWithBiasAndAddActivation& matched,
    std::vector<bool>* invalidated_nodes, std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);
  DCHECK(IsConvOrMatMul(contraction));
  const NodeDef& activation = graph->node(matched.activation);
  const NodeDef& bias_add = graph->node(matched.bias_add);
  const NodeDef& add = graph->node(matched.add);

  zendnnInfo(ZENDNN_FWKLOG, "Fuse ", contraction.op(), " with BiasAdd ",
             bias_add.op(), " and Add ", add.op(), " and Activation ",
             activation.op(), ": activation=", activation.name(),
             " bias_add=", bias_add.name(), " add=", add.name(),
             " contraction=", contraction.name());

  NodeDef fused_node;
  fused_node.set_name(activation.name());

  fused_node.set_device(contraction.device());
  fused_node.add_input(contraction.input(0));  // 0: input
  fused_node.add_input(contraction.input(1));  // 1: filter
  fused_node.add_input(bias_add.input(1));     // 2: bias

  // Add OP has two inputs, one is conv+bias pattern matched previously,
  // the other input to add is fused here.
  fused_node.add_input(add.input(1 - matched.port_id));

  if (IsConv2D(contraction)) {
    fused_node.set_op(kFusedConv2D);
    CopyConv2DAttributes(contraction, &fused_node);
  } else if (IsDepthwiseConv2dNative(contraction)) {
    fused_node.set_op(kFusedDepthwiseConv2dNative);
    CopyDepthwiseConv2dNativeAttributes(contraction, &fused_node);
    // TODO(plugin) : Check if _ZenFusedBatchMatMul is a possibility.
  } else {
    CHECK(false);
  }

  SetFusedOpAttributesWithActivation(&fused_node, &activation,
                                     {"BiasAdd", "Add"}, 2);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_node), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.activation] = true;
  (*nodes_to_delete)[matched.add] = true;
  (*nodes_to_delete)[matched.bias_add] = true;
  (*nodes_to_delete)[matched.contraction] = true;

  return OkStatus();
}

Status AddFusedConv2DNode(RemapperContext* ctx,
                          const ContractionWithBatchNorm& matched,
                          std::vector<bool>* invalidated_nodes,
                          std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);
  DCHECK(IsConv2D(contraction)) << "Only Conv2D supported for now";
  const NodeDef& fused_batch_norm = graph->node(matched.fused_batch_norm);
  zendnnInfo(ZENDNN_FWKLOG, "Fuse Conv2D with BatchNorm: batch_norm =",
             fused_batch_norm.name(), " conv2d =", contraction.name());

  NodeDef fused_node;
  fused_node.set_name(fused_batch_norm.name());
  fused_node.set_op(kFusedConv2D);
  fused_node.set_device(contraction.device());
  fused_node.add_input(contraction.input(0));       // 0: input
  fused_node.add_input(contraction.input(1));       // 1: filter
  fused_node.add_input(fused_batch_norm.input(1));  // 2: scale
  fused_node.add_input(fused_batch_norm.input(2));  // 3: offset
  fused_node.add_input(fused_batch_norm.input(3));  // 4: mean
  fused_node.add_input(fused_batch_norm.input(4));  // 5: variance

  AddInputShapesAttr(*ctx, matched.contraction);
  CopyConv2DAttributes(contraction, &fused_node);
  SetFusedOpAttributes(&fused_node, {"FusedBatchNorm"},
                       /*num_args=*/4, /*epsilon=*/matched.epsilon);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_node), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.fused_batch_norm] = true;
  (*nodes_to_delete)[matched.contraction] = true;

  return OkStatus();
}

Status AddFusedConv2DNode(RemapperContext* ctx,
                          const ContractionWithBatchNormAndActivation& matched,
                          std::vector<bool>* invalidated_nodes,
                          std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);

  DCHECK(IsConv2D(contraction)) << "Only Conv2D supported for now";

  const NodeDef& activation = graph->node(matched.activation);
  const NodeDef& fused_batch_norm = graph->node(matched.fused_batch_norm);
  zendnnInfo(ZENDNN_FWKLOG, "Fuse Conv2D with BatchNorm and ", activation.op(),
             ": activation =", activation.name(),
             " batch_norm =", fused_batch_norm.name(),
             " conv2d =", contraction.name());

  NodeDef fused_node;
  fused_node.set_name(activation.name());
  fused_node.set_op(kFusedConv2D);
  fused_node.set_device(contraction.device());
  fused_node.add_input(contraction.input(0));       // 0: input
  fused_node.add_input(contraction.input(1));       // 1: filter
  fused_node.add_input(fused_batch_norm.input(1));  // 2: scale
  fused_node.add_input(fused_batch_norm.input(2));  // 3: offset
  fused_node.add_input(fused_batch_norm.input(3));  // 4: mean
  fused_node.add_input(fused_batch_norm.input(4));  // 5: variance

  AddInputShapesAttr(*ctx, matched.contraction);
  CopyConv2DAttributes(contraction, &fused_node, &activation);
  SetFusedOpAttributes(&fused_node, {"FusedBatchNorm", activation.op()},
                       /*num_args=*/4, /*epsilon=*/matched.epsilon);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_node), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.activation] = true;
  (*nodes_to_delete)[matched.contraction] = true;
  (*nodes_to_delete)[matched.fused_batch_norm] = true;

  return OkStatus();
}

Status AddFusedBatchNormExNode(RemapperContext* ctx,
                               const FusedBatchNormEx& matched,
                               std::vector<bool>* invalidated_nodes,
                               std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& fused_batch_norm = graph->node(matched.fused_batch_norm);
  const NodeDef& activation = graph->node(matched.activation);

  zendnnInfo(ZENDNN_FWKLOG, "Fuse ", activation.op(),
             " with FusedBatchNorm: activation=", activation.name(),
             " for fused_batch_norm=", fused_batch_norm.name());

  // Replace FusedBatchNorm with _FusedBatchNormEx + Activation.
  NodeDef fused_op;
  fused_op.set_op(kFusedBatchNormEx);
  fused_op.set_name(fused_batch_norm.name());
  fused_op.set_device(fused_batch_norm.device());

  fused_op.add_input(fused_batch_norm.input(0));  // 0: input
  fused_op.add_input(fused_batch_norm.input(1));  // 1: scale
  fused_op.add_input(fused_batch_norm.input(2));  // 2: offset
  fused_op.add_input(fused_batch_norm.input(3));  // 3: estimated_mean
  fused_op.add_input(fused_batch_norm.input(4));  // 4: estimated_var

  CopyFusedBatchNormAttributes(fused_batch_norm, &fused_op);

  auto* attrs = fused_op.mutable_attr();
  SetAttrValue(activation.op(), &(*attrs)["activation_mode"]);

  AddNodeAttr("num_side_inputs", 0, &fused_op);

  // Turn activation node into Identity node.
  NodeDef identity_op;
  identity_op.set_op("Identity");
  identity_op.set_name(activation.name());
  identity_op.set_device(fused_batch_norm.device());
  identity_op.add_input(fused_batch_norm.name());
  (*identity_op.mutable_attr())["T"] = attrs->at("T");

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_op), &status);
  TF_ABORT_IF_ERROR(status);
  mutation->AddNode(std::move(identity_op), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.fused_batch_norm] = true;
  (*invalidated_nodes)[matched.activation] = true;

  return OkStatus();
}

Status AddPadWithContractionNode(RemapperContext* ctx,
                                 const PadWithContraction& matched,
                                 std::vector<bool>* invalidated_nodes,
                                 std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& pad = graph->node(matched.pad);
  const auto& pad_view = ctx->graph_view.GetNode(matched.pad);
  const auto& contraction_view = ctx->graph_view.GetNode(matched.contraction);
  const NodeDef& contraction = graph->node(matched.contraction);

  string padding;
  TF_CHECK_OK(GetNodeAttr(contraction, "padding", &padding));

  // Get original explicit padding values if padding = EXPLICIT.
  std::vector<int32> explicit_paddings_orig = {};
  if (padding == "EXPLICIT") {
    TF_CHECK_OK(GetNodeAttr(pad, "explicit_paddings", &explicit_paddings_orig));
  }

  // Index 0 has the input data and Index 1 has the padding values (which is
  // needed).
  const auto& const_pad_val_node_view =
      pad_view->GetRegularFanin(1).node_view();
  const auto const_pad_val_node_def = const_pad_val_node_view->node();
  Tensor explicit_padding_tensor;
  std::vector<int32> explicit_paddings;
  if (explicit_padding_tensor.FromProto(
          const_pad_val_node_def->attr().at("value").tensor())) {
    // Number of elements in explicit_padding_tensor (should be 8).
    int length = explicit_padding_tensor.NumElements();
    // 'padding_1d_tensor' is an Eigen Tensor with datatype int32.
    auto padding_1d_tensor = explicit_padding_tensor.flat<int32>();
    // For dimension i (starting from 0), the padding values
    // will be at 2*i and 2*i + 1.
    for (int index_pad = 0; index_pad < length; index_pad++) {
      if (padding == "VALID") {
        explicit_paddings.insert(explicit_paddings.begin() + index_pad,
                                 padding_1d_tensor(index_pad));
      } else if (padding == "EXPLICIT") {
        explicit_paddings.insert(explicit_paddings.begin() + index_pad,
                                 padding_1d_tensor(index_pad) +
                                     explicit_paddings_orig.at(index_pad));
      }
    }
  }

  auto* conv2d_mutable_attr = contraction_view->node()->mutable_attr();
  SetAttrValue("EXPLICIT", &(*conv2d_mutable_attr)["padding"]);
  SetAttrValue(explicit_paddings, &(*conv2d_mutable_attr)["explicit_paddings"]);

  NodeDef pad_with_conv;
  pad_with_conv.set_name(contraction.name());
  pad_with_conv.set_device(contraction.device());
  pad_with_conv.add_input(pad.input(0));          // 0: input
  pad_with_conv.add_input(contraction.input(1));  // 1: filter
  // Add bias input if contraction is _FusedConv2D/_FusedDepthwiseConv2dNative.
  if (contraction.op() == kFusedConv2D ||
      contraction.op() == kFusedDepthwiseConv2dNative) {
    pad_with_conv.add_input(contraction.input(2));  // 2: bias
  }

  if (IsConv2D(contraction) || contraction.op() == kFusedConv2D) {
    pad_with_conv.set_op(contraction.op());
    CopyConv2DAttributes(contraction, &pad_with_conv);
  } else if (IsDepthwiseConv2dNative(contraction) ||
             contraction.op() == kFusedDepthwiseConv2dNative) {
    pad_with_conv.set_op(contraction.op());
    CopyDepthwiseConv2dNativeAttributes(contraction, &pad_with_conv);
  } else {
    CHECK(false);
  }
  if (HasNodeAttr(contraction, "fused_ops")) {
    std::vector<string> fused_ops;
    // Only bias is allowed with fused contraction.
    int num_args = 1;
    float epsilon = 0.0;
    TF_CHECK_OK(GetNodeAttr(contraction, "fused_ops", &fused_ops));
    auto* attr = pad_with_conv.mutable_attr();
    SetAttrValue(fused_ops, &(*attr)["fused_ops"]);
    SetAttrValue(num_args, &(*attr)["num_args"]);
    SetAttrValue(epsilon, &(*attr)["epsilon"]);
  }

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(pad_with_conv), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.contraction] = true;
  (*nodes_to_delete)[matched.pad] = true;

  return OkStatus();
}

// Contraction + Mul(scale).
Status AddFusedContractionNode(RemapperContext* ctx,
                               const ContractionWithMul& matched,
                               std::vector<bool>* invalidated_nodes,
                               std::vector<bool>* nodes_to_delete) {
  const GraphDef* graph = ctx->graph_view.graph();
  const NodeDef& contraction = graph->node(matched.contraction);
  const NodeDef& mul = graph->node(matched.mul);
  const NodeDef& scalar = graph->node(matched.scalar);

  NodeDef fused_op;
  fused_op.set_name(mul.name());
  fused_op.set_device(contraction.device());
  fused_op.add_input(contraction.input(0));  // 0: input
  fused_op.add_input(contraction.input(1));  // 1: filter
  fused_op.add_input(scalar.name());         // 2: scale
  fused_op.set_op(kFusedBatchMatMulV2);

  CopyBatchMatMulAttributes(contraction, &fused_op);
  SetFusedOpAttributes(&fused_op, {kBinaryMul}, 1);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_op), &status);
  TF_ABORT_IF_ERROR(status);
  TF_ABORT_IF_ERROR(mutation->Apply());

  (*invalidated_nodes)[matched.mul] = true;
  (*nodes_to_delete)[matched.contraction] = true;
  return OkStatus();
}

Status AddFusedMatMulBiasAddAndGelu(
    RemapperContext* ctx, const std::map<string, int>& matched_nodes_map,
    const std::set<int>& remove_node_indices,
    std::vector<bool>* invalidated_nodes, std::vector<bool>* nodes_to_delete,
    bool is_gelu_approximate, bool expand_dims) {
  auto* output_node =
      ctx->graph_view.GetNode(matched_nodes_map.at("output"))->node();
  auto* matmul_node =
      ctx->graph_view.GetNode(matched_nodes_map.at("matmul"))->node();

  NodeDef fused_node;
  // Fused node should have the name of terminal node of the fusion.
  fused_node.set_name(output_node->name());
  fused_node.set_op("_FusedMatMul");
  fused_node.set_device(matmul_node->device());
  fused_node.add_input(matmul_node->input(0));
  fused_node.add_input(matmul_node->input(1));
  if (is_gelu_approximate || expand_dims) {
    fused_node.add_input(matmul_node->input(2));
  } else {
    auto* bias_add_node =
        ctx->graph_view.GetNode(matched_nodes_map.at("bias_add"))->node();
    fused_node.add_input(bias_add_node->input(1));
  }
  CopyMatMulAttributes(*matmul_node, &fused_node);
  if (is_gelu_approximate) {
    SetFusedOpAttributes(&fused_node, {"BiasAdd", "GeluApproximate"});
  } else {
    SetFusedOpAttributes(&fused_node, {"BiasAdd", "GeluExact"});
  }
  if (expand_dims) {
    auto* attr = fused_node.mutable_attr();
    SetAttrValue(true, &(*attr)["is_reshape"]);
  }

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_node), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());
  (*invalidated_nodes)[matched_nodes_map.at("output")] = true;

  for (const auto& node_idx : remove_node_indices) {
    (*nodes_to_delete)[node_idx] = true;
  }
  return OkStatus();
}

Status AddFusedBatchMatMul(RemapperContext* ctx,
                           const std::map<string, int>& matched_nodes_map,
                           const std::set<int>& remove_node_indices,
                           const std::vector<string>& input_node_names,
                           std::vector<bool>* invalidated_nodes,
                           std::vector<bool>* nodes_to_delete) {
  auto* output_node =
      ctx->graph_view.GetNode(matched_nodes_map.at("output"))->node();
  auto* batch_matmul_node =
      ctx->graph_view.GetNode(matched_nodes_map.at("batch_matmul"))->node();

  NodeDef fused_node;
  fused_node.set_name(output_node->name());
  // Note: The op and kernel definition for the "_FusedBatchMatMulV2" op is not
  // present. We are sure that it will be rewritten with
  // "_ZenFusedBatchMatMulV2" from zen layout pass.
  fused_node.set_op(kFusedBatchMatMulV2);

  fused_node.set_device(batch_matmul_node->device());
  for (const auto& name : input_node_names) fused_node.add_input(name);

  CopyBatchMatMulAttributes(*batch_matmul_node, &fused_node);
  SetFusedOpAttributes(&fused_node, {kBinaryMul, kAdd}, /*num_args=*/2);

  utils::Mutation* mutation = ctx->graph_view.GetMutationBuilder();
  Status status;
  mutation->AddNode(std::move(fused_node), &status);
  TF_RETURN_IF_ERROR(status);
  TF_RETURN_IF_ERROR(mutation->Apply());
  (*invalidated_nodes)[matched_nodes_map.at("output")] = true;

  for (const auto& node_idx : remove_node_indices) {
    (*nodes_to_delete)[node_idx] = true;
  }
  return OkStatus();
}

}  // namespace

Status RunRemapper(const char* device_name, const GrapplerItem& item,
                   const GraphDef& graph_def, GraphDef* optimized_graph) {
  Status status;
  GraphDef multable_graph_def = graph_def;
  RemapperContext ctx(item, &multable_graph_def, &status,
                      /*level*/ RemapperLevel::BASIC);

  // Processing graph in reverse-topological sorted order allows to remap
  // longer chains of dependent ops in one pass.
  TF_RETURN_IF_ERROR(
      ctx.graph_view.SortTopologically(/*ignore_cycles=*/false, {}));

  const int num_nodes = multable_graph_def.node_size();
  // Skip nodes that were invalidated by a remapper, e.g. do not process BiasAdd
  // and Activation nodes that were fused into a Conv2D node.
  std::vector<bool> invalidated_nodes(num_nodes);
  std::vector<bool> nodes_to_delete(num_nodes);

  zendnnInfo(ZENDNN_FWKLOG, "RemapperPass: Start to fuse nodes.");

  // Infer statically first and only once.
  ctx.GetGraphProperties();

  bool is_visited = false;
  string last_op;
  for (int i = num_nodes - 1; i >= 0;) {
    NodeDef* node_def = (ctx.graph_view.GetNode(i))->node();

    // IMPORTANT: Always keep this dynamic check in the start.
    // Dynamic check node status:
    //   1. Do normal fusion check when current node is visited first time.
    //   2. Recheck this node only if it's new fused and never rechecked before.
    //   3. Iterate to next node after current node is visited and not fused, or
    //      already rechecked.
    if (is_visited) {
      if (invalidated_nodes[i] && last_op != node_def->op()) {
        // Recheck current node to find more possible fusion.
        zendnnInfo(ZENDNN_FWKLOG, "Recheck node ", node_def->op(), " : ",
                   node_def->name());
        last_op = node_def->op();
      } else {
        // Iterate to next node and reset all flags.
        --i;
        is_visited = false;
        last_op = node_def->op();
        continue;
      }
    } else {
      last_op = node_def->op();
      is_visited = true;
    }

    // Check if node was deleted by one of the previous remaps.
    if (nodes_to_delete[i]) {
      continue;
    }
    zendnnInfo(ZENDNN_FWKLOG, " Processing ", node_def->op(), " ",
               node_def->name());
    {
      // Keras Dense layer fwd fusion.
      KerasDenseLayerFwd keras_dense_layer_fwd;
      if (FindKerasDenseLayerFwd(ctx, i, &keras_dense_layer_fwd)) {
        zendnnInfo(ZENDNN_FWKLOG, " Found KerasDenseLayerFwd pattern.");
        TF_ABORT_IF_ERROR(AddKerasDenseLayerFwd(
            &ctx, keras_dense_layer_fwd, &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // Remap Conv2D+BiasAdd+Add+Activation into the _FusedConv2D.
      ContractionWithBiasAndAddActivation contract_with_bias_and_add_activation;
      if (FindContractionWithBiasAndAddActivation(
              ctx, i, &contract_with_bias_and_add_activation)) {
        zendnnInfo(ZENDNN_FWKLOG,
                   " Found ContractionWithBiasAndAddActivation pattern.");
        TF_ABORT_IF_ERROR(
            AddFusedContractionNode(&ctx, contract_with_bias_and_add_activation,
                                    &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // Remap MatMul+Relu into the _FusedMatMul.
      ContractionWithActivation contract_with_activation;
      if (FindContractionWithActivation(ctx, i, &contract_with_activation)) {
        zendnnInfo(ZENDNN_FWKLOG, " Found ContractionWithActivation pattern.");
        AddFusedContractionNode(&ctx, contract_with_activation,
                                &invalidated_nodes, &nodes_to_delete);
        continue;
      }

      // Remap _FusedMatMul{MatMul + BiasAdd} + Sigmoid into the _FusedMatMul.
      ContractionWithActivation contract_with_sigmoid;
      if (FindContractionWithSigmoid(ctx, i, &contract_with_sigmoid)) {
        zendnnInfo(ZENDNN_FWKLOG, " Found ContractionWithSigmoid pattern.");
        AddFusedMatMulSigmoidNode(&ctx, contract_with_sigmoid,
                                  &invalidated_nodes, &nodes_to_delete);
        continue;
      }

      // Remap Conv2D+BiasAdd+Add into the _FusedConv2D.
      ContractionWithBiasAddAndAdd contract_with_bias_and_add;
      if (FindContractionWithBiasAddAndAdd(ctx, i,
                                           &contract_with_bias_and_add)) {
        zendnnInfo(ZENDNN_FWKLOG,
                   " Found ContractionWithBiasAddAndAdd pattern.");
        TF_ABORT_IF_ERROR(
            AddFusedContractionNode(&ctx, contract_with_bias_and_add,
                                    &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // Remap {Conv2D,DepthwiseConv2D,MatMul}+BiasAdd into the
      // _Fused{Conv2D,DepthwiseConv2dNative,MatMul}.
      ContractionWithBiasAdd contract_with_bias;
      if (FindContractionWithBias(ctx, i, &contract_with_bias)) {
        zendnnInfo(ZENDNN_FWKLOG, " Found ContractionWithBiasAdd pattern.");
        TF_ABORT_IF_ERROR(AddFusedContractionNode(
            &ctx, contract_with_bias, &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // Remap {Conv2D,DepthwiseConv2D,MatMul}+BiasAdd+Activation into
      // the _Fused{Conv2D,DepthwiseConv2dNative,MatMul}.
      ContractionWithBiasAddAndActivation contract_with_bias_and_activation;
      if (FindContractionWithBiasAndActivation(
              ctx, i, &contract_with_bias_and_activation)) {
        zendnnInfo(ZENDNN_FWKLOG,
                   " Found ContractionWithBiasAddAndActivation pattern.");
        TF_RETURN_IF_ERROR(
            AddFusedContractionNode(&ctx, contract_with_bias_and_activation,
                                    &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // Remap Conv2D+FusedBatchNorm into the _FusedConv2D.
      ContractionWithBatchNorm contract_with_batch_norm;
      if (FindConv2DWithBatchNorm(ctx, i, &contract_with_batch_norm)) {
        zendnnInfo(ZENDNN_FWKLOG, " Found ContractionWithBatchNorm pattern.");
        TF_RETURN_IF_ERROR(AddFusedConv2DNode(&ctx, contract_with_batch_norm,
                                              &invalidated_nodes,
                                              &nodes_to_delete));
        continue;
      }

      // Remap Conv2D+FusedBatchNorm+Activation into the _FusedConv2D.
      ContractionWithBatchNormAndActivation
          contract_with_batch_norm_and_activation;
      if (FindConv2DWithBatchNormAndActivation(
              ctx, i, &contract_with_batch_norm_and_activation)) {
        zendnnInfo(ZENDNN_FWKLOG,
                   " Found ContractionWithBatchNormAndActivation pattern.");
        TF_RETURN_IF_ERROR(
            AddFusedConv2DNode(&ctx, contract_with_batch_norm_and_activation,
                               &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // Remap FusedBatchNorm+<Activation> into _FusedBatchNormEx.
      FusedBatchNormEx fused_batch_norm_ex;
      if (FindFusedBatchNormEx(ctx, i, &fused_batch_norm_ex)) {
        zendnnInfo(ZENDNN_FWKLOG, " Found FindFusedBatchNormEx pattern.");
        TF_ABORT_IF_ERROR(AddFusedBatchNormExNode(
            &ctx, fused_batch_norm_ex, &invalidated_nodes, &nodes_to_delete));
      }

      // Remap BatchMatMul + Mul into the _FusedBatchMatMul.
      ContractionWithMul contract_with_mul;
      if (FindContractionWithMul(ctx, i, &contract_with_mul)) {
        zendnnInfo(ZENDNN_FWKLOG, " Found BatchMatMul pattern with BinaryMul.");
        AddFusedContractionNode(&ctx, contract_with_mul, &invalidated_nodes,
                                &nodes_to_delete);
        continue;
      }

      // Remap MatMul + BiasAdd + gelu-subgraph.
      std::map<string, int> matched_nodes_map;
      std::set<int> remove_node_indices;
      bool is_gelu_approximate = false;
      bool expand_dims = false;
      if (FindMatMulBiasAddAndGelu(&ctx, i, &matched_nodes_map,
                                   &remove_node_indices, &is_gelu_approximate,
                                   &expand_dims)) {
        zendnnInfo(ZENDNN_FWKLOG, " Found MatMulBiasAddAndGelu pattern with",
                   (is_gelu_approximate ? "" : "out"), " Gelu Approximate.");
        TF_ABORT_IF_ERROR(AddFusedMatMulBiasAddAndGelu(
            &ctx, matched_nodes_map, remove_node_indices, &invalidated_nodes,
            &nodes_to_delete, is_gelu_approximate, expand_dims));
        continue;
      }

      // Remap BatchMatMul + Mul + AddV2 into the _FusedBatchMatMul.
      matched_nodes_map.clear();
      remove_node_indices.clear();
      std::vector<string> input_node_names;
      input_node_names.clear();
      if (FindFusedBatchMatMul(&ctx, i, &matched_nodes_map,
                               &remove_node_indices, &input_node_names)) {
        zendnnInfo(ZENDNN_FWKLOG,
                   " Found BatchMatMul pattern with BinaryMul and Add.");
        TF_RETURN_IF_ERROR(AddFusedBatchMatMul(
            &ctx, matched_nodes_map, remove_node_indices, input_node_names,
            &invalidated_nodes, &nodes_to_delete));
        continue;
      }

      // Remap Pad + (_Fused)Conv2D to (_Fused)Conv2D.
      PadWithContraction pad_conv;
      if (FindPadWithContraction(ctx, i, &pad_conv)) {
        zendnnInfo(ZENDNN_FWKLOG, " Found PadWithContraction pattern.");
        TF_ABORT_IF_ERROR(AddPadWithContractionNode(
            &ctx, pad_conv, &invalidated_nodes, &nodes_to_delete));
        continue;
      }
    }
  }

  // Remove invalidated nodes.
  utils::Mutation* mutation = ctx.graph_view.GetMutationBuilder();
  for (int i = 0; i < num_nodes; ++i) {
    if (nodes_to_delete[i]) {
      mutation->RemoveNode(ctx.graph_view.GetNode(i));
    }
  }
  TF_ABORT_IF_ERROR(mutation->Apply());

  *optimized_graph = std::move(multable_graph_def);
  return OkStatus();
}

}  // namespace graph
}  // namespace amd_cpu_plugin
