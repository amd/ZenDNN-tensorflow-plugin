/*******************************************************************************
 * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 ******************************************************************************/

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_GRAPH_AUTO_MIXED_PRECISION_AUTO_MIXED_PRECISION_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_GRAPH_AUTO_MIXED_PRECISION_AUTO_MIXED_PRECISION_H_

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow_plugin/src/amd_cpu/graph/utils/grappler_item.h"
#include "tensorflow_plugin/src/amd_cpu/graph/utils/utils.h"
#include "tensorflow_plugin/src/amd_cpu/util/tf_buffer.h"

namespace amd_cpu_plugin {
namespace graph {

enum class AutoMixedPrecisionMode { CPU_BFLOAT16 };

Status RunAutoMixedPrecision(const char* device_name, const GrapplerItem& item,
                             const GraphDef& graph_def, GraphDef* output);

}  // namespace graph
}  // namespace amd_cpu_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_GRAPH_AUTO_MIXED_PRECISION_AUTO_MIXED_PRECISION_H_
