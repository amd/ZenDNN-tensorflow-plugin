/*******************************************************************************
 * Modifications Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights
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

#include "tensorflow_plugin/src/amd_cpu/util/attr_value_util.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/strings/escaping.h"
#include "protos/attr_value.pb.h"
#include "protos/tensor.pb.h"
#include "protos/tensor_shape.pb.h"
#include "protos/types.pb.h"
#include "tensorflow_plugin/src/amd_cpu/util/errors.h"
#include "tensorflow_plugin/src/amd_cpu/util/hash.h"
#include "tensorflow_plugin/src/amd_cpu/util/plugin_tensor.h"
#include "tensorflow_plugin/src/amd_cpu/util/proto_serialization.h"
#include "tensorflow_plugin/src/amd_cpu/util/protobuf.h"
#include "tensorflow_plugin/src/amd_cpu/util/str_util.h"
#include "tensorflow_plugin/src/amd_cpu/util/stringpiece.h"
#include "tensorflow_plugin/src/amd_cpu/util/types.h"

namespace amd_cpu_plugin {
namespace {

// Do not construct large tensors to compute their hash or compare for equality.
constexpr int kMaxAttrValueTensorByteSize = 32 * 1024 * 1024;  // 32mb

string SummarizeString(const string& str) {
  string escaped = absl::CEscape(str);

  // If the string is long, replace the middle with ellipses.
  constexpr int kMaxStringSummarySize = 80;
  if (escaped.size() >= kMaxStringSummarySize) {
    StringPiece prefix(escaped);
    StringPiece suffix = prefix;
    prefix.remove_suffix(escaped.size() - 10);
    suffix.remove_prefix(escaped.size() - 10);
    return strings::StrCat("\"", prefix, "...", suffix, "\"");
  } else {
    return strings::StrCat("\"", escaped, "\"");
  }
}

string SummarizeTensor(const TensorProto& tensor_proto) {
  Tensor t;
  if (!t.FromProto(tensor_proto)) {
    return strings::StrCat(
        "<Invalid TensorProto: ", tensor_proto.ShortDebugString(), ">");
  }
  return t.DebugString();
}

string SummarizeFunc(const NameAttrList& func) {
  std::vector<string> entries;
  for (const auto& p : func.attr()) {
    entries.push_back(
        strings::StrCat(p.first, "=", SummarizeAttrValue(p.second)));
  }
  std::sort(entries.begin(), entries.end());
  return strings::StrCat(func.name(), "[", absl::StrJoin(entries, ", "), "]");
}

const char* EnumName_DataType(::amd_cpu_plugin::DataType value) {
  switch (value) {
    case 0:
      return "DT_INVALID";
    case 1:
      return "DT_FLOAT";
    case 2:
      return "DT_DOUBLE";
    case 3:
      return "DT_INT32";
    case 4:
      return "DT_UINT8";
    case 5:
      return "DT_INT16";
    case 6:
      return "DT_INT8";
    case 7:
      return "DT_STRING";
    case 8:
      return "DT_COMPLEX64";
    case 9:
      return "DT_INT64";
    case 10:
      return "DT_BOOL";
    case 11:
      return "DT_QINT8";
    case 12:
      return "DT_QUINT8";
    case 13:
      return "DT_QINT32";
    case 14:
      return "DT_BFLOAT16";
    case 15:
      return "DT_QINT16";
    case 16:
      return "DT_QUINT16";
    case 17:
      return "DT_UINT16";
    case 18:
      return "DT_COMPLEX128";
    case 19:
      return "DT_HALF";
    case 20:
      return "DT_RESOURCE";
    case 21:
      return "DT_VARIANT";
    case 22:
      return "DT_UINT32";
    case 23:
      return "DT_UINT64";
    case 101:
      return "DT_FLOAT_REF";
    case 102:
      return "DT_DOUBLE_REF";
    case 103:
      return "DT_INT32_REF";
    case 104:
      return "DT_UINT8_REF";
    case 105:
      return "DT_INT16_REF";
    case 106:
      return "DT_INT8_REF";
    case 107:
      return "DT_STRING_REF";
    case 108:
      return "DT_COMPLEX64_REF";
    case 109:
      return "DT_INT64_REF";
    case 110:
      return "DT_BOOL_REF";
    case 111:
      return "DT_QINT8_REF";
    case 112:
      return "DT_QUINT8_REF";
    case 113:
      return "DT_QINT32_REF";
    case 114:
      return "DT_BFLOAT16_REF";
    case 115:
      return "DT_QINT16_REF";
    case 116:
      return "DT_QUINT16_REF";
    case 117:
      return "DT_UINT16_REF";
    case 118:
      return "DT_COMPLEX128_REF";
    case 119:
      return "DT_HALF_REF";
    case 120:
      return "DT_RESOURCE_REF";
    case 121:
      return "DT_VARIANT_REF";
    case 122:
      return "DT_UINT32_REF";
    case 123:
      return "DT_UINT64_REF";
    default:
      return "";
  }
}

}  // namespace

string SummarizeAttrValue(const AttrValue& attr_value) {
  switch (attr_value.value_case()) {
    case AttrValue::kS:
      return SummarizeString(attr_value.s());
    case AttrValue::kI:
      return strings::StrCat(attr_value.i());
    case AttrValue::kF:
      return strings::StrCat(attr_value.f());
    case AttrValue::kB:
      return attr_value.b() ? "true" : "false";
    case AttrValue::kType:
      return EnumName_DataType(attr_value.type());
    case AttrValue::kShape:
      return PartialTensorShape::DebugString(attr_value.shape());
    case AttrValue::kTensor:
      return SummarizeTensor(attr_value.tensor());
    case AttrValue::kList: {
      std::vector<string> pieces;
      if (attr_value.list().s_size() > 0) {
        for (int i = 0; i < attr_value.list().s_size(); ++i) {
          pieces.push_back(SummarizeString(attr_value.list().s(i)));
        }
      } else if (attr_value.list().i_size() > 0) {
        for (int i = 0; i < attr_value.list().i_size(); ++i) {
          pieces.push_back(strings::StrCat(attr_value.list().i(i)));
        }
      } else if (attr_value.list().f_size() > 0) {
        for (int i = 0; i < attr_value.list().f_size(); ++i) {
          pieces.push_back(strings::StrCat(attr_value.list().f(i)));
        }
      } else if (attr_value.list().b_size() > 0) {
        for (int i = 0; i < attr_value.list().b_size(); ++i) {
          pieces.push_back(attr_value.list().b(i) ? "true" : "false");
        }
      } else if (attr_value.list().type_size() > 0) {
        for (int i = 0; i < attr_value.list().type_size(); ++i) {
          pieces.push_back(EnumName_DataType(attr_value.list().type(i)));
        }
      } else if (attr_value.list().shape_size() > 0) {
        for (int i = 0; i < attr_value.list().shape_size(); ++i) {
          pieces.push_back(
              TensorShape::DebugString(attr_value.list().shape(i)));
        }
      } else if (attr_value.list().tensor_size() > 0) {
        for (int i = 0; i < attr_value.list().tensor_size(); ++i) {
          pieces.push_back(SummarizeTensor(attr_value.list().tensor(i)));
        }
      } else if (attr_value.list().func_size() > 0) {
        for (int i = 0; i < attr_value.list().func_size(); ++i) {
          pieces.push_back(SummarizeFunc(attr_value.list().func(i)));
        }
      }
      constexpr int kMaxListSummarySize = 50;
      if (pieces.size() >= kMaxListSummarySize) {
        pieces.erase(pieces.begin() + 5, pieces.begin() + (pieces.size() - 6));
        pieces[5] = "...";
      }
      return strings::StrCat("[", absl::StrJoin(pieces, ", "), "]");
    }
    case AttrValue::kFunc: {
      return SummarizeFunc(attr_value.func());
    }
    case AttrValue::kPlaceholder:
      return strings::StrCat("$", attr_value.placeholder());
    case AttrValue::VALUE_NOT_SET:
      return "<Unknown AttrValue type>";
  }
  return "<Unknown AttrValue type>";  // Prevent missing return warning
}

Status AttrValueHasType(const AttrValue& attr_value, StringPiece type) {
  int num_set = 0;

#define VALIDATE_FIELD(name, type_string, oneof_case)                         \
  do {                                                                        \
    if (attr_value.has_list()) {                                              \
      if (attr_value.list().name##_size() > 0) {                              \
        if (type != "list(" type_string ")") {                                \
          return errors::InvalidArgument(                                     \
              "AttrValue had value with type 'list(" type_string ")' when '", \
              type, "' expected");                                            \
        }                                                                     \
        ++num_set;                                                            \
      }                                                                       \
    } else if (attr_value.value_case() == AttrValue::oneof_case) {            \
      if (type != type_string) {                                              \
        return errors::InvalidArgument(                                       \
            "AttrValue had value with type '" type_string "' when '", type,   \
            "' expected");                                                    \
      }                                                                       \
      ++num_set;                                                              \
    }                                                                         \
  } while (false)

  VALIDATE_FIELD(s, "string", kS);
  VALIDATE_FIELD(i, "int", kI);
  VALIDATE_FIELD(f, "float", kF);
  VALIDATE_FIELD(b, "bool", kB);
  VALIDATE_FIELD(type, "type", kType);
  VALIDATE_FIELD(shape, "shape", kShape);
  VALIDATE_FIELD(tensor, "tensor", kTensor);
  VALIDATE_FIELD(func, "func", kFunc);

#undef VALIDATE_FIELD

  if (attr_value.value_case() == AttrValue::kPlaceholder) {
    return errors::InvalidArgument(
        "AttrValue had value with unexpected type 'placeholder'");
  }

  // If the attr type is 'list', we expect attr_value.has_list() to be
  // true.  However, proto3's attr_value.has_list() can be false when
  // set to an empty list for GraphDef versions <= 4. So we simply
  // check if has_list is false and some other field in attr_value is
  // set to flag the error.  This test can be made more strict once
  // support for GraphDef versions <= 4 is dropped.
  if (absl::StartsWith(type, "list(") && !attr_value.has_list()) {
    if (num_set) {
      return errors::InvalidArgument(
          "AttrValue missing value with expected type '", type, "'");
    } else {
      // Indicate that we have a list, but an empty one.
      ++num_set;
    }
  }

  // Okay to have an empty list, but not to be missing a non-list value.
  if (num_set == 0 && !absl::StartsWith(type, "list(")) {
    return errors::InvalidArgument(
        "AttrValue missing value with expected type '", type, "'");
  }

  // Ref types and DT_INVALID are illegal, and DataTypes must
  // be a valid enum type.
  if (type == "type") {
    if (!DataType_IsValid(attr_value.type())) {
      return errors::InvalidArgument("AttrValue has invalid DataType enum: ",
                                     attr_value.type());
    }
    if (IsRefType(attr_value.type())) {
      return errors::InvalidArgument(
          "AttrValue must not have reference type value of ",
          DataTypeString(attr_value.type()));
    }
    if (attr_value.type() == DT_INVALID) {
      return errors::InvalidArgument("AttrValue has invalid DataType");
    }
  } else if (type == "list(type)") {
    for (auto as_int : attr_value.list().type()) {
      const DataType dtype = static_cast<DataType>(as_int);
      if (!DataType_IsValid(dtype)) {
        return errors::InvalidArgument("AttrValue has invalid DataType enum: ",
                                       as_int);
      }
      if (IsRefType(dtype)) {
        return errors::InvalidArgument(
            "AttrValue must not have reference type value of ",
            DataTypeString(dtype));
      }
      if (dtype == DT_INVALID) {
        return errors::InvalidArgument("AttrValue contains invalid DataType");
      }
    }
  }

  return OkStatus();
}

void SetAttrValue(const AttrValue& value, AttrValue* out) { *out = value; }

#define DEFINE_SET_ATTR_VALUE_ONE(ARG_TYPE, FIELD) \
  void SetAttrValue(ARG_TYPE value, AttrValue* out) { out->set_##FIELD(value); }

#define DEFINE_SET_ATTR_VALUE_LIST(ARG_TYPE, FIELD)                       \
  void SetAttrValue(ARG_TYPE value, AttrValue* out) {                     \
    out->mutable_list()->Clear(); /* create list() even if value empty */ \
    for (const auto& v : value) {                                         \
      out->mutable_list()->add_##FIELD(v);                                \
    }                                                                     \
  }

#define DEFINE_SET_ATTR_VALUE_BOTH(ARG_TYPE, FIELD) \
  DEFINE_SET_ATTR_VALUE_ONE(ARG_TYPE, FIELD)        \
  DEFINE_SET_ATTR_VALUE_LIST(gtl::ArraySlice<ARG_TYPE>, FIELD)

DEFINE_SET_ATTR_VALUE_ONE(const string&, s)
DEFINE_SET_ATTR_VALUE_LIST(gtl::ArraySlice<string>, s)
DEFINE_SET_ATTR_VALUE_BOTH(const char*, s)
DEFINE_SET_ATTR_VALUE_BOTH(int64, i)
DEFINE_SET_ATTR_VALUE_BOTH(int32, i)
DEFINE_SET_ATTR_VALUE_BOTH(float, f)
DEFINE_SET_ATTR_VALUE_BOTH(double, f)
DEFINE_SET_ATTR_VALUE_BOTH(bool, b)
DEFINE_SET_ATTR_VALUE_LIST(const std::vector<bool>&, b)
DEFINE_SET_ATTR_VALUE_LIST(std::initializer_list<bool>, b)
DEFINE_SET_ATTR_VALUE_BOTH(DataType, type)

void SetAttrValue(const tstring& value, AttrValue* out) {
  out->set_s(value.data(), value.size());
}

void SetAttrValue(gtl::ArraySlice<tstring> value, AttrValue* out) {
  out->mutable_list()->Clear();
  for (const auto& v : value) {
    out->mutable_list()->add_s(v.data(), v.size());
  }
}

void SetAttrValue(StringPiece value, AttrValue* out) {
  out->set_s(value.data(), value.size());
}

void SetAttrValue(const gtl::ArraySlice<StringPiece> value, AttrValue* out) {
  out->mutable_list()->Clear();  // Create list() even if value empty.
  for (const auto& v : value) {
    out->mutable_list()->add_s(v.data(), v.size());
  }
}

void MoveAttrValue(std::vector<string>&& value, AttrValue* out) {
  out->mutable_list()->Clear();  // Create list() even if value empty.
  for (auto& v : value) {
    out->mutable_list()->add_s(std::move(v));
  }
}

void SetAttrValue(const TensorShape& value, AttrValue* out) {
  value.AsProto(out->mutable_shape());
}

void SetAttrValue(const TensorShapeProto& value, AttrValue* out) {
  *out->mutable_shape() = value;
}

void SetAttrValue(const PartialTensorShape& value, AttrValue* out) {
  value.AsProto(out->mutable_shape());
}

void SetAttrValue(const gtl::ArraySlice<TensorShape> value, AttrValue* out) {
  out->mutable_list()->Clear();  // Create list() even if value empty.
  for (const auto& v : value) {
    v.AsProto(out->mutable_list()->add_shape());
  }
}

void SetAttrValue(gtl::ArraySlice<TensorShapeProto> value, AttrValue* out) {
  out->mutable_list()->Clear();  // Create list() even if value empty.
  for (const auto& v : value) {
    *out->mutable_list()->add_shape() = v;
  }
}

void SetAttrValue(const gtl::ArraySlice<PartialTensorShape> value,
                  AttrValue* out) {
  out->mutable_list()->Clear();  // Create list() even if value empty.
  for (const auto& v : value) {
    v.AsProto(out->mutable_list()->add_shape());
  }
}

void SetAttrValue(const TensorProto& value, AttrValue* out) {
  *out->mutable_tensor() = value;
}

void SetAttrValue(const gtl::ArraySlice<TensorProto> value, AttrValue* out) {
  out->mutable_list()->Clear();  // Create list() even if value empty.
  for (const auto& v : value) {
    *out->mutable_list()->add_tensor() = v;
  }
}

void SetAttrValue(const NameAttrList& value, AttrValue* out) {
  *out->mutable_func() = value;
}

void SetAttrValue(gtl::ArraySlice<NameAttrList> value, AttrValue* out) {
  out->mutable_list()->Clear();  // Create list() even if value empty.
  for (const auto& v : value) {
    *out->mutable_list()->add_func() = v;
  }
}

bool HasPlaceHolder(const AttrValue& val) {
  switch (val.value_case()) {
    case AttrValue::kList: {
      for (const NameAttrList& func : val.list().func()) {
        for (const auto& p : func.attr()) {
          if (HasPlaceHolder(p.second)) {
            return true;
          }
        }
      }
      break;
    }
    case AttrValue::kFunc:
      for (const auto& p : val.func().attr()) {
        if (HasPlaceHolder(p.second)) {
          return true;
        }
      }
      break;
    case AttrValue::kPlaceholder:
      return true;
    default:
      break;
  }
  return false;
}

bool SubstitutePlaceholders(const SubstituteFunc& substitute,
                            AttrValue* value) {
  switch (value->value_case()) {
    case AttrValue::kList: {
      for (NameAttrList& func : *value->mutable_list()->mutable_func()) {
        for (auto& p : *func.mutable_attr()) {
          if (!SubstitutePlaceholders(substitute, &p.second)) {
            return false;
          }
        }
      }
      break;
    }
    case AttrValue::kFunc:
      for (auto& p : *(value->mutable_func()->mutable_attr())) {
        if (!SubstitutePlaceholders(substitute, &p.second)) {
          return false;
        }
      }
      break;
    case AttrValue::kPlaceholder:
      return substitute(value->placeholder(), value);
    case AttrValue::VALUE_NOT_SET:
      return false;
    default:
      break;
  }
  return true;
}

}  // namespace amd_cpu_plugin
