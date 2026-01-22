/*******************************************************************************
 * Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 ******************************************************************************/

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_BATCH_MATMUL_UTIL_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_BATCH_MATMUL_UTIL_H_

#include <cstdint>
#include <vector>

#include "tensorflow_plugin/src/amd_cpu/util/tensor_shape.h"

namespace amd_cpu_plugin {

using dims = std::vector<int64_t>;

// This function adjusts the dimensions of an input tensor to match the
// dimensions of an output tensor.
inline void ExpandInputDimsToOutputShape(const TensorShape &input_shape,
                                         const TensorShape &output_shape,
                                         dims *reshaped_dims) {
  auto ndims_input = input_shape.dims();
  auto ndims_output = output_shape.dims();
  auto dim_offset = ndims_output - ndims_input;
  DCHECK(dim_offset > 0);
  reshaped_dims->clear();
  reshaped_dims->resize(ndims_output, 1);
  auto input_dims = input_shape.dim_sizes();
  for (int dim_idx = 0; dim_idx < ndims_input; ++dim_idx) {
    reshaped_dims->at(dim_idx + dim_offset) = input_dims[dim_idx];
  }
}

// Extracts dimensions from a TensorFlow shape into a vector of int64_t.
inline std::vector<int64_t> ExtractDimsFromTFShape(const TensorShape &shape) {
  std::vector<int64_t> dims;
  for (int i = 0; i < shape.dims(); ++i) {
    dims.push_back(shape.dim_size(i));
  }
  return dims;
}

}  // namespace amd_cpu_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_BATCH_MATMUL_UTIL_H_
