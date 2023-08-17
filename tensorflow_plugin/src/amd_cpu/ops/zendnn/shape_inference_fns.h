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

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_OPS_ZENDNN_SHAPE_INFERENCE_FNS_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_OPS_ZENDNN_SHAPE_INFERENCE_FNS_H_

void unknown_shape_fn(TF_ShapeInferenceContext* ctx, TF_Status* status);

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_OPS_ZENDNN_SHAPE_INFERENCE_FNS_H_
