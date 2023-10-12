/*******************************************************************************
 * Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 ******************************************************************************/

/* Copyright (c) 2021-2022 Intel Corporation

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

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_GRAPH_UTILS_GRAPPLER_ITEM_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_GRAPH_UTILS_GRAPPLER_ITEM_H_

#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/experimental/grappler/grappler.h"
#include "tensorflow_plugin/src/amd_cpu/util/errors.h"
#include "tensorflow_plugin/src/amd_cpu/util/status.h"

namespace amd_cpu_plugin {
namespace graph {

class GrapplerItem {
 public:
  explicit GrapplerItem(const TF_GrapplerItem* tf_item);
  TF_GrapplerItem* GetTfGrapplerItem() const { return item_; }
  std::unordered_set<string> NodesToPreserve() const;
  std::vector<string> fetch;

 private:
  TF_GrapplerItem* item_;
};

}  // namespace graph
}  // namespace amd_cpu_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_GRAPH_UTILS_GRAPPLER_ITEM_H_
