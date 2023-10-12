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

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_GRAPH_UTILS_FUNCTION_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_GRAPH_UTILS_FUNCTION_H_

#include <memory>
#include <string>

#include "protos/graph.pb.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/experimental/grappler/grappler.h"
#include "tensorflow_plugin/src/amd_cpu/util/errors.h"
#include "tensorflow_plugin/src/amd_cpu/util/gtl/flatmap.h"
#include "tensorflow_plugin/src/amd_cpu/util/mutex.h"
#include "tensorflow_plugin/src/amd_cpu/util/status.h"

namespace amd_cpu_plugin {
namespace graph {

class FunctionLibraryDefinition {
 public:
  explicit FunctionLibraryDefinition(const GraphDef& g_def);
  ~FunctionLibraryDefinition();
  Status LookUpOpDef(const std::string& op_type_name, OpDef* op_def) const;
  const FunctionDef* Find(const std::string& func) const TF_LOCKS_EXCLUDED(mu_);

 private:
  TF_FunctionLibraryDefinition* func_;

  struct FunctionDefAndOpRegistration {
    explicit FunctionDefAndOpRegistration(const FunctionDef& fdef_in);

    const FunctionDef fdef;
  };

  std::shared_ptr<FunctionDefAndOpRegistration> FindHelper(
      const string& func) const TF_SHARED_LOCKS_REQUIRED(mu_);

  mutable mutex mu_;
  gtl::FlatMap<string, std::shared_ptr<FunctionDefAndOpRegistration>>
      function_defs_ TF_GUARDED_BY(mu_);
  gtl::FlatMap<string, string> func_grad_ TF_GUARDED_BY(mu_);
};

}  // namespace graph
}  // namespace amd_cpu_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_GRAPH_UTILS_FUNCTION_H_
