/*******************************************************************************
 * Modifications Copyright (c) 2024 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 ******************************************************************************/

/* Copyright (c) 2022 Intel Corporation

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

#include "tensorflow_plugin/src/amd_cpu/graph/config_util.h"

#include <cstring>

namespace amd_cpu_plugin {
namespace {
ConfigProto& Configs() {
  static ConfigProto config;
  return config;
}
}  // namespace

void zen_set_config(const ConfigProto& config) { Configs() = config; }

ConfigProto zen_get_config() { return Configs(); }

bool isxehpc_value;
ConfigProto zen_get_isxehpc() {
  ConfigProto isxehpc_proto;
  GraphOptions* isxehpc_graph = isxehpc_proto.mutable_graph_options();
  isxehpc_graph->set_device_isxehpc(isxehpc_value);
  return isxehpc_proto;
}

}  // namespace amd_cpu_plugin
