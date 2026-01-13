/*******************************************************************************
 * Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 ******************************************************************************/

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_ZENDNNL_UTILS_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_ZENDNNL_UTILS_H_

#include <cstdlib>
#include <string>

#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_kernel_common.h"

namespace amd_cpu_plugin {

// Log ZenDNNL success
inline void LogZenDNNLSuccess(const char* kernel_name) {
  // std::cout << "ZenDNNL: Successfully executed " << kernel_name << " kernel"
  // << std::endl;
}

// Log ZenDNNL fallback to ZenDNN
inline void LogZenDNNLFallback(const char* kernel_name, const char* reason) {
  // std::cout << "ZenDNNL: " << kernel_name << " execution " << reason
  //           << ", falling back to ZenDNN implementation" << std::endl;
}

// Log ZenDNNL initialization
inline void LogZenDNNLInfo(const char* kernel_name, const char* message) {
  // std::cout << "ZenDNNL " << kernel_name << ": " << message << std::endl;
}

// Check if ZenDNN MatMul Direct API is enabled via environment variable
inline bool IsZenDnnMatmulDirectEnabled() {
  static int use_direct = -1;  // Cache the result
  if (use_direct == -1) {
    const char* env_value = std::getenv("USE_ZENDNN_MATMUL_DIRECT");
    use_direct = env_value ? std::atoi(env_value) : 0;
  }
  return use_direct != 0;
}

}  // namespace amd_cpu_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_ZENDNNL_UTILS_H_
