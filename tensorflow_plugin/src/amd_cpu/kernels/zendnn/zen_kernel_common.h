/*******************************************************************************
 * Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_KERNEL_COMMON_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_KERNEL_COMMON_H_

#include "tensorflow_plugin/src/amd_cpu/util/op_kernel.h"

namespace amd_cpu_plugin {

// Common parameters for ZenDNN nodes.
struct ZendnnParameters {
  bool reorder_before = false;
  bool reorder_after = false;
  bool is_eager = false;
  int in_links = 0;
  int out_links = 0;
  bool reset = false;
};

// Initializes and validates ZenDNN parameters configured
// by OpKernel attributes.
//
// @input context Context from which prameters are read
//         params Parameters for ZenDNN Op
// @return Status
inline Status InitZendnnParameters(OpKernelConstruction* context,
                                   ZendnnParameters* params) {
  TF_RETURN_IF_ERROR(
      context->GetAttr("reorder_before", &params->reorder_before));
  TF_RETURN_IF_ERROR(context->GetAttr("reorder_after", &params->reorder_after));
  TF_RETURN_IF_ERROR(context->GetAttr("is_eager", &params->is_eager));
  TF_RETURN_IF_ERROR(context->GetAttr("in_links", &params->in_links));
  TF_RETURN_IF_ERROR(context->GetAttr("out_links", &params->out_links));
  TF_RETURN_IF_ERROR(context->GetAttr("reset", &params->reset));

  return OkStatus();
}

}  // namespace amd_cpu_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_KERNEL_COMMON_H_
