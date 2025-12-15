/*******************************************************************************
 * Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 ******************************************************************************/

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_ZENDNNL_UTILS_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_ZENDNNL_UTILS_H_

#include <cstdlib>
#include <string>

#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_kernel_common.h"

namespace amd_cpu_plugin {

// Check if ZenDNNL is enabled via environment variable
inline bool IsZenDNNLEnabled() {
  static int use_zendnnl = -1;  // Cache the result
  if (use_zendnnl == -1) {
    const char* env_value = std::getenv("USE_ZENDNNL");
    use_zendnnl = env_value ? std::atoi(env_value) : 0;
  }
  return use_zendnnl != 0;
}

// Log ZenDNNL success
inline void LogZenDNNLSuccess(const char* kernel_name) {
  zendnnInfo(ZENDNN_FWKLOG, "ZenDNNL: Successfully executed ", kernel_name,
             " kernel");
}

// Log ZenDNNL fallback to ZenDNN
inline void LogZenDNNLFallback(const char* kernel_name, const char* reason) {
  zendnnInfo(ZENDNN_FWKLOG, "ZenDNNL: ", kernel_name, " execution ", reason,
             ", falling back to ZenDNN implementation");
}

// Log ZenDNNL initialization
inline void LogZenDNNLInfo(const char* kernel_name, const char* message) {
  zendnnInfo(ZENDNN_FWKLOG, "ZenDNNL ", kernel_name, ": ", message);
}

}  // namespace amd_cpu_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_ZENDNNL_UTILS_H_
