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
 ******************************************************************************/

#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_mempool.h"

#include <map>

namespace amd_cpu_plugin {

// For protecting ThreadID Map
std::mutex mtx;
// Map for storing TF thread id
std::map<std::thread::id, unsigned int> tf_thread_id_map;

// initialize memory pool static array for use by the kernels
// declared in zen_mempool.h
ZenMemoryPoolBase *ZenMemoryPoolBase::zen_memory_pool_arr_[ZEN_MEM_POOL_LIMIT] =
    {NULL};
int ZenMemoryPoolBase::zen_memory_pool_count_ = 0;

DataType GetDataTypeFromMacro(ZenTensorType type) {
  DataType data_type = DT_FLOAT;
  switch (type) {
    case ZenTensorType::kQint8:
      data_type = DT_QINT8;
      break;
    case ZenTensorType::kQuint8:
      data_type = DT_QUINT8;
      break;
    case ZenTensorType::kBfloat16:
      data_type = DT_BFLOAT16;
      break;
    case ZenTensorType::kFloat:
      data_type = DT_FLOAT;
      break;
  }
  return data_type;
}

// This function takes thread id (comes from TF threadpool) and coverts into
// integer thread ID using Map.
// Same integer thread ID is used for creating seperate Memory pool for inter_op
// threads.
unsigned int GetZenTFthreadId(std::thread::id thread_id) {
  static unsigned int num_threads = 0;
  unsigned int int_id = -1;
  std::map<std::thread::id, unsigned int>::iterator it;

  it = tf_thread_id_map.find(thread_id);
  if (it != tf_thread_id_map.end()) {
    int_id = tf_thread_id_map[thread_id];
  } else {
    mtx.lock();
    tf_thread_id_map[thread_id] = num_threads;
    int_id = tf_thread_id_map[thread_id];
    num_threads++;
    mtx.unlock();
  }
  return int_id;
}

}  // namespace amd_cpu_plugin
