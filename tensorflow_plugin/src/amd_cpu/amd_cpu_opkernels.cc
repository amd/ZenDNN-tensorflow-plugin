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

// TensorFlow C API headers
#include "tensorflow/c/kernels.h"
// TensorFlow plugin headers
#include "tensorflow_plugin/src/amd_cpu/graph/cpu_optimizer.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_kernels_init.h"
#include "tensorflow_plugin/src/amd_cpu/ops/zendnn/zen_ops_init.h"

void TF_InitKernel() {
  RegisterZenOps();
  RegisterZenKernels();
}

void TF_InitGraph(TP_OptimizerRegistrationParams* params, TF_Status* status) {
  params->struct_size = TP_OPTIMIZER_REGISTRATION_PARAMS_STRUCT_SIZE;
  params->optimizer_configs->struct_size = TP_OPTIMIZER_CONFIGS_STRUCT_SIZE;
  params->optimizer->struct_size = TP_OPTIMIZER_STRUCT_SIZE;

  params->optimizer_configs->layout_optimizer = TF_TriState_Off;
  params->optimizer_configs->auto_mixed_precision = TF_TriState_Off;
  params->optimizer_configs->auto_mixed_precision_mkl = TF_TriState_Off;

  // Set functions to create a new optimizer.
  params->optimizer->optimize_func =
      (amd_cpu_plugin::graph::Optimizer_Optimize);
  params->optimizer->destroy_func = (amd_cpu_plugin::graph::Optimizer_Destroy);

  params->device_type = "CPU";  // amd_cpu_plugin::DEVICE_CPU;
  params->optimizer->create_func = (amd_cpu_plugin::graph::Optimizer_Create);
}
