/*******************************************************************************
 * Modifications Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 ******************************************************************************/

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow_plugin/src/amd_cpu/util/op_def_util.h"

#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow_plugin/src/amd_cpu/util/attr_value_util.h"
#include "tensorflow_plugin/src/amd_cpu/util/errors.h"
#include "tensorflow_plugin/src/amd_cpu/util/gtl/map_util.h"
#include "tensorflow_plugin/src/amd_cpu/util/hash.h"
#include "tensorflow_plugin/src/amd_cpu/util/proto_serialization.h"
#include "tensorflow_plugin/src/amd_cpu/util/protobuf.h"
#include "tensorflow_plugin/src/amd_cpu/util/scanner.h"
#include "tensorflow_plugin/src/amd_cpu/util/str_util.h"
#include "tensorflow_plugin/src/amd_cpu/util/strcat.h"
#include "tensorflow_plugin/src/amd_cpu/util/stringpiece.h"
#include "tensorflow_plugin/src/amd_cpu/util/types.h"

namespace amd_cpu_plugin {

const OpDef::AttrDef* FindAttr(StringPiece name, const OpDef& op_def) {
  for (int i = 0; i < op_def.attr_size(); ++i) {
    if (op_def.attr(i).name() == name) {
      return &op_def.attr(i);
    }
  }
  return nullptr;
}

OpDef::AttrDef* FindAttrMutable(StringPiece name, OpDef* op_def) {
  for (int i = 0; i < op_def->attr_size(); ++i) {
    if (op_def->attr(i).name() == name) {
      return op_def->mutable_attr(i);
    }
  }
  return nullptr;
}

const OpDef::ArgDef* FindInputArg(StringPiece name, const OpDef& op_def) {
  for (int i = 0; i < op_def.input_arg_size(); ++i) {
    if (op_def.input_arg(i).name() == name) {
      return &op_def.input_arg(i);
    }
  }
  return nullptr;
}

const ApiDef::Arg* FindInputArg(StringPiece name, const ApiDef& api_def) {
  for (int i = 0; i < api_def.in_arg_size(); ++i) {
    if (api_def.in_arg(i).name() == name) {
      return &api_def.in_arg(i);
    }
  }
  return nullptr;
}

#define VALIDATE(EXPR, ...)                                        \
  do {                                                             \
    if (!(EXPR)) {                                                 \
      return errors::InvalidArgument(                              \
          __VA_ARGS__, "; in OpDef: ", op_def.ShortDebugString()); \
    }                                                              \
  } while (false)

bool IsValidOpName(StringPiece sp) {
  using ::amd_cpu_plugin::strings::Scanner;

  Scanner scanner(sp);
  scanner.One(Scanner::UPPERLETTER).Any(Scanner::LETTER_DIGIT_UNDERSCORE);

  while (true) {
    if (!scanner.GetResult())  // Some error in previous iteration.
      return false;
    if (scanner.empty())  // No error, but nothing left, good.
      return true;

    // Absorb another name/namespace, starting with a '>'
    scanner.One(Scanner::RANGLE)
        .One(Scanner::UPPERLETTER)
        .Any(Scanner::LETTER_DIGIT_UNDERSCORE);
  }
}

#undef VALIDATE

namespace {

// Returns true if every element of `sub` is contained in `super`.
template <class T>
bool IsSubsetOf(const T& sub, const T& super) {
  for (const auto& o : sub) {
    bool found = false;
    for (const auto& n : super) {
      if (o == n) {
        found = true;
        break;
      }
    }
    if (!found) return false;
  }
  return true;
}

}  // namespace

void RemoveNonDeprecationDescriptionsFromOpDef(OpDef* op_def) {
  for (int i = 0; i < op_def->input_arg_size(); ++i) {
    op_def->mutable_input_arg(i)->clear_description();
  }
  for (int i = 0; i < op_def->output_arg_size(); ++i) {
    op_def->mutable_output_arg(i)->clear_description();
  }
  for (int i = 0; i < op_def->attr_size(); ++i) {
    op_def->mutable_attr(i)->clear_description();
  }
  op_def->clear_summary();
  op_def->clear_description();
}

void RemoveDescriptionsFromOpDef(OpDef* op_def) {
  RemoveNonDeprecationDescriptionsFromOpDef(op_def);
  if (op_def->has_deprecation()) {
    op_def->mutable_deprecation()->clear_explanation();
  }
}

void RemoveDescriptionsFromOpList(OpList* op_list) {
  for (int i = 0; i < op_list->op_size(); ++i) {
    OpDef* op_def = op_list->mutable_op(i);
    RemoveDescriptionsFromOpDef(op_def);
  }
}

}  // namespace amd_cpu_plugin
