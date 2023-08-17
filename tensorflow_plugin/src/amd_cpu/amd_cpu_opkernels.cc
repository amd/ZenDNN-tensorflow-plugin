/*******************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/

// TensorFlow C API headers
#include "tensorflow/c/kernels.h"
// TensorFlow plugin headers
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_kernels_init.h"
#include "tensorflow_plugin/src/amd_cpu/ops/zendnn/zen_ops_init.h"

void TF_InitKernel() {
  RegisterZenOps();
  RegisterZenKernels();
}
