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

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_MEMPOOL_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_MEMPOOL_H_

#include <sys/types.h>

#include <mutex>   // NOLINT(build/c++11)
#include <thread>  // NOLINT(build/c++11)

#include "tensorflow_plugin/src/amd_cpu/util/op_kernel.h"
#include "tensorflow_plugin/src/amd_cpu/util/tensor_format.h"
#include "tensorflow_plugin/src/amd_cpu/util/zen_utils.h"

// This is for supporting future integration with streams.
// TODO(zendnn): Test with maximum no of possible streams and tune
//               ZEN_MEM_POOL_LIMIT accordingly.
#define ZEN_MEM_POOL_LIMIT 64

// ZEN_TENSOR_POOL_LIMIT defines the limit for active tensors inside pool for
// given Memory pool.
// TODO(zendnn): Test with increased limit and tune it accordingly.
#define ZEN_TENSOR_POOL_LIMIT 32

// ZEN_TENSOR_SIZE_FACTOR increased the max size required for storing the max
// o/p of graph. Currently with CNN and NLP(Bert, WideDeep, DLRM) models, this
// is fixed with 1.
// TODO(zendnn): Test with other models and see if its required and tune it
//               acordingly.
// #define ZEN_TENSOR_SIZE_FACTOR 1

using zendnn::zendnn_getenv_int;

namespace amd_cpu_plugin {

extern std::mutex mtx;
unsigned int GetZenTFthreadId(std::thread::id thread_id);

enum class ZenTensorType { kQint8 = 0, kQuint8 = 1, kFloat = 2, kBfloat16 = 3 };

DataType GetDataTypeFromMacro(ZenTensorType type);

// ZenTensorPool structure holds zen_tensor_ptr with its state 0, -1 and > 0
// Possible states  =-1   (not allocated),
//                  = 0   (allocated and free)
//                  > 0   (occupied, no. of links with other node)
// zen_tensor_size is size of pointed memory (no. of elements inside tensor).
typedef struct PersistentTensorState {
  Tensor *zen_tensor_ptr;
  void *raw_buff;
  int zen_tensor_ptr_status;
  uint64_t zen_tensor_size;
  ZenTensorType zen_type;
} ZenTensorPool;

// Class ZenMemoryPoolBase defines a description for memory pool with all tensor
// pointer.
class ZenMemoryPoolBase {
 public:
  // zen_memory_pool_arr_ holds number of memory pool that exist. In case of
  // multiple streams, each stream will have its own memory pool. Currently
  // ZEN_MEM_POOL_LIMIT is the limit, can be made dynamic in future for
  // supporting [streams > ZEN_MEM_POOL_LIMIT]. Single Memory pool object will
  // be created for each stream, every call to GetZenMemPool(<fixed index>) will
  // return same object.
  // zen_memory_pool_count_ holds the no of active memory pool.
  static int zen_memory_pool_count_;
  static ZenMemoryPoolBase *zen_memory_pool_arr_[ZEN_MEM_POOL_LIMIT];
  // zen_tensor_pool_arr_ will hold all ptr and state of tensors created in the
  // pool.
  // zen_tensor_ptr_status will hold the state of those tensor.
  // Possible states =-1   (not allocated),
  //                 = 0    (allocated and free)
  //                 > 0    (tensor links with other node)
  ZenTensorPool *zen_tensor_pool_arr_;

  // No. of allocated tensors inside pool.
  unsigned int zen_tensor_pool_size_;

  // Max limit for active tensors inside pool.
  unsigned int zen_tensor_pool_limit_;

  // zen_enable_mempool_ holds the type of memory pool.
  // Possible states 0(disabled),
  //                 1(mempool at graph level, memory footprint > 0 < 2)
  //                 2(mempool at node level, memory footprint > 0 and 1)
  unsigned int zen_enable_mempool_;

  // max_size_enable_ will allocate all tensor in pool of size equal to size
  // of the o/p first layer or running max with pool array.
  // TODO(zendnn): calulate max_size as part of cost function during graph
  // analysis phase and then use it accordingly.
  bool max_size_enable_;

  // Max shape of allocated tensors in the pool.
  TensorShape max_shape_;

  // Reset TensorPool after every graph execution.
  bool zen_tensor_pool_reset_;
};

template <class T>
class ZenMemoryPool : public ZenMemoryPoolBase {
 private:
  // Initialize pool object with default values.
  ZenMemoryPool() {
    zen_tensor_pool_size_ = 0;
    max_shape_ = TensorShape();
    zen_tensor_pool_reset_ = false;
    zen_tensor_pool_arr_ = NULL;

    zendnnEnv zen_env_obj = readEnv();
    zen_enable_mempool_ = zen_env_obj.zenEnableMemPool;

    // To enable/disable reduced memory pool(default) or fixed memory pool
    // tensor from env variable. Reduced memory pool tensor works with different
    // size tensors in pool. Usually size of output tensor size go down as we go
    // deeper into model. Some models are exception to this. For those, in case
    // of Reduced memory pool, some of the layers will use default memory
    // allocation once we hit the pool limit with ZEN_TENSOR_POOL_LIMIT.
    // Otherwise works with Fixed memory pool tensor, this works with finding
    // max from increasing size as we go deeper into model. In future as a part
    // of cost function max size will be calculated and used accordingly.
    max_size_enable_ = zendnn_getenv_int("ZENDNN_TENSOR_BUF_MAXSIZE_ENABLE");

    // Getting max pool limit from env variable.
    // If ZENDNN_TENSOR_POOL_LIMIT env variable is not defined, use default
    // value for ZEN_TENSOR_POOL_LIMIT.
    zen_tensor_pool_limit_ =
        zendnn_getenv_int("ZENDNN_TENSOR_POOL_LIMIT", ZEN_TENSOR_POOL_LIMIT);
    zen_tensor_pool_limit_ =
        (zen_tensor_pool_limit_ <= 0) ? 1 : zen_tensor_pool_limit_;

    zen_tensor_pool_arr_ = static_cast<ZenTensorPool *>(
        malloc(zen_tensor_pool_limit_ * sizeof(ZenTensorPool)));

    for (int i = 0; i < zen_tensor_pool_limit_; i++) {
      zen_tensor_pool_arr_[i].zen_tensor_ptr = NULL;
      zen_tensor_pool_arr_[i].raw_buff = NULL;
      zen_tensor_pool_arr_[i].zen_tensor_ptr_status = -1;
      zen_tensor_pool_arr_[i].zen_tensor_size = 0;
      zen_tensor_pool_arr_[i].zen_type = ZenTensorType::kQint8;
    }
  }

  // Destroy Memory pool once done with usage.
  ~ZenMemoryPool() {
    for (int i = 0; i < zen_tensor_pool_size_; i++) {
      delete zen_tensor_pool_arr_[i].zen_tensor_ptr;
    }
    free(zen_tensor_pool_arr_);
  }

 public:
  // Get Memory pool pointer from Global array of memory pool based on index.
  // Create ZenMemPool object, if not created corresponding to that index.
  static ZenMemoryPool *GetZenMemPool(const int index) {
    // ZEN_MEM_POOL_LIMIT is the hard limit on the total no. of ZenMemoryPool
    // TODO(zendnn): Need to tune ZEN_MEM_POOL_LIMIT based on the available
    // memory or make it grow dynamically
    if (index >= ZEN_MEM_POOL_LIMIT) {
      return NULL;
    }
    mtx.lock();
    if (!zen_memory_pool_arr_[index]) {
      zen_memory_pool_arr_[index] = new ZenMemoryPool();
      zen_memory_pool_count_++;
    }
    mtx.unlock();
    return static_cast<ZenMemoryPool<T> *>(zen_memory_pool_arr_[index]);
  }

  // Free zen_memory_pool_arr_ based on index passed.
  static void FreeZenMemPool(const int index) {
    if (index >= ZEN_MEM_POOL_LIMIT) {
      return;
    }
    mtx.lock();
    if (zen_memory_pool_arr_[index]) {
      delete zen_memory_pool_arr_[index];
      zen_memory_pool_count_--;
    }
    mtx.unlock();
  }

  // Reset status of all Tensor as free at the start of graph execution.
  void ResetPoolStatus() {
    for (int i = 0; i < zen_tensor_pool_size_; i++) {
      zen_tensor_pool_arr_[i].zen_tensor_ptr_status = 0;
    }
  }

  uint64_t GetTensorSize(const TensorShape &shape) {
    uint64_t size = 1;
    int num_dimensions = shape.dims();
    for (int i = 0; i < num_dimensions; i++) {
      size *= shape.dim_size(i);
    }
    return size;
  }

  // Acquire Tensor buffer from the given pool object. If pool is not
  // initialized or buffer is not free, create PERSISTENT tensor and add to the
  // pool.
  int AcquireZenPoolTensor(OpKernelContext *context, Tensor **output,
                           TensorShape out_shape, int outlinks, bool reset,
                           ZenTensorType type, int out_index = 0) {
    if (reset && zen_tensor_pool_size_) {
      zen_tensor_pool_reset_ = true;
    }

    /*
    // TODO(zendnn): Compute max_size as part of cost function during graph
    // analysis phase. Once we supoort above cost function, do early return
    // based on below condition.
    if (max_size_enable_) {
      // Fall back to default tensor allocation if required, when out_shape is
      // more than max_shape_ of pool.
      uint64_t out_size = GetTensorSize(out_shape);
      uint64_t max_size = GetTensorSize(max_shape_);
      if (out_size > max_size) {
        zendnnInfo(ZENDNN_FWKLOG, " TF-MEM-POOL: Requested Tensor from
              ZenMemPool, but falling back to default allocation as out_size(",
              out_size, ") > max_size(", max_size, ")of Pool\n");
        return 1;
      }
    }
    */

    bool acquire_flag = false;
    bool free_flag = false;

    // Search for free tensor in pool based on zen_enable_mempool_, tensor_size
    // and out_size.
    for (int i = 0; i < zen_tensor_pool_size_; i++) {
      if (zen_tensor_pool_arr_[i].zen_tensor_ptr_status == 0) {
        free_flag = true;

        // Go to next free tensor when out_size is more than tensor_size of pool
        // at given offset.
        uint64_t out_size = GetTensorSize(out_shape);
        uint64_t tensor_size = zen_tensor_pool_arr_[i].zen_tensor_size;
        if (out_size > tensor_size ||
            (zen_enable_mempool_ == 2 && out_size < tensor_size)) {
          continue;
        }
        *output = (zen_tensor_pool_arr_[i].zen_tensor_ptr);
        zen_tensor_pool_arr_[i].zen_type = type;
        zen_tensor_pool_arr_[i].raw_buff =
            static_cast<T *>((*output)->template flat<T>().data());
        zen_tensor_pool_arr_[i].zen_tensor_ptr_status = outlinks;
        (*output)->set_shape(out_shape);
        if (out_index >= 0) {
          context->set_output(out_index, **output);
        }
        acquire_flag = true;
        zendnnInfo(ZENDNN_FWKLOG, "TF-MEM-POOL: Acquired TensorPool Ptr[", i,
                   "] pointed to size(no. of elements)", tensor_size);

        break;
      }
    }

    // If requested tensor not found in pool, go ahead and create new tensor
    // inside pool.
    if (!acquire_flag) {
      if (zen_tensor_pool_size_ == zen_tensor_pool_limit_) {
        if (free_flag) {
          zendnnInfo(ZENDNN_FWKLOG,
                     "TF-MEM-POOL: Requested Tensor from ZenMemPool, but "
                     "falling back to default allocation as out_size > "
                     "tensor_size available inside Pool.");
        } else {
          zendnnInfo(
              ZENDNN_FWKLOG,
              "TF-MEM-POOL: Requested Tensor from ZenMemPool, but "
              "falling back to default allocation as zen_tensor_pool_size_ "
              "== ZEN_TENSOR_POOL_LIMIT");
        }
        return 1;
      }

      unsigned int pool_offset = zen_tensor_pool_size_;
      TensorShape shape;

      // Set max_shape_ based on current layer's output dimension and
      // ZEN_TENSOR_SIZE_FACTOR. Most of the cases Output dimension goes down
      // after first layer. However few models are exception to this.
      // The max_size required can be computed during first run for graph
      // execution and same can be used for allocation. But this will not give
      // optimal performance for first graph execution.
      // TODO(zendnn): Compute max_size as part of cost function during graph
      // analysis phase.
      uint64_t out_size = GetTensorSize(out_shape);
      uint64_t max_size = GetTensorSize(max_shape_);
      if (out_size > max_size) {
        max_shape_ = out_shape;
      }

      // max_size_enable_ creates all tensor with increasing size inside the
      // pool.
      if (max_size_enable_) {
        uint64_t max_size = GetTensorSize(max_shape_);
        zen_tensor_pool_arr_[pool_offset].zen_tensor_size = max_size;
        shape = max_shape_;
      } else {
        uint64_t size = GetTensorSize(out_shape);
        zen_tensor_pool_arr_[pool_offset].zen_tensor_size = size;
        shape = out_shape;
      }

      zen_tensor_pool_arr_[pool_offset].zen_tensor_ptr = new Tensor();
      DataType data_type = GetDataTypeFromMacro(type);
      context->allocate_temp(data_type, shape,
                             zen_tensor_pool_arr_[pool_offset].zen_tensor_ptr);

      *output = (zen_tensor_pool_arr_[pool_offset].zen_tensor_ptr);
      zen_tensor_pool_arr_[pool_offset].raw_buff =
          static_cast<T *>((*output)->template flat<T>().data());

      zen_tensor_pool_arr_[pool_offset].zen_tensor_ptr_status = outlinks;
      (*output)->set_shape(out_shape);
      if (out_index >= 0) {
        context->set_output(out_index, **output);
      }
      acquire_flag = true;
      zen_tensor_pool_size_++;
      zendnnInfo(ZENDNN_FWKLOG,
                 "TF-MEM-POOL: Allocation done for Tensor in Pool of size = ",
                 (*output)->TotalBytes() / sizeof(T), " elements",
                 " zenTensorPoolCount = ", zen_tensor_pool_size_ - 1);
      zendnnInfo(ZENDNN_FWKLOG, "TF-MEM-POOL: Acquired TensorPool Ptr[",
                 pool_offset, "] pointed to size(no. of elements)",
                 (*output)->TotalBytes() / sizeof(T));
    }
    return 0;
  }

  // This will update the state of Memory pool by decrementing status of zen
  // pool array based on the input buffer comparison.
  void ZenMemPoolFree(OpKernelContext *context, void *input) {
    if (zen_enable_mempool_ == 1) {
      // This block optimizes the buffer reuse across mempool(each inter op has
      // its own pool memory). Currently this has some performance issues.
      // TODO(zendnn): Fix this.
      mtx.lock();
      for (int i = 0; i < zen_memory_pool_count_; i++) {
        if (zen_memory_pool_arr_[i]) {
          for (int j = 0; j < zen_memory_pool_arr_[i]->zen_tensor_pool_size_;
               j++) {
            void *output_array =
                zen_memory_pool_arr_[i]->zen_tensor_pool_arr_[j].raw_buff;
            if (input == output_array) {
              zen_memory_pool_arr_[i]
                  ->zen_tensor_pool_arr_[j]
                  .zen_tensor_ptr_status--;
              break;
            }
          }
        }
      }
      mtx.unlock();
    }

    // This will be enabled, when we reset pool after last zen node execution.
    if (zen_tensor_pool_reset_) {
      ResetPoolStatus();
      zen_tensor_pool_reset_ = false;
    }
  }

  // Method to update the 'use status' of buffer from tensor pool. Basically it
  // resets the 'use status' with the status value received as argument.
  // Currently this method is used in convolution fused sum optimization where
  // the input buffer is re-used as output buffer.
  void ZenMemPoolUpdateTensorPtrStatus(OpKernelContext *context, void *input,
                                       int status, bool reset) {
    if (zen_enable_mempool_ == 1) {
      // This block optimizes the buffer reuse across mempool (each inter op has
      // its own pool memory). Currently this has some performance issues.
      // TODO(zendnn): Fix this.
      mtx.lock();
      for (int i = 0; i < zen_memory_pool_count_; i++) {
        if (zen_memory_pool_arr_[i]) {
          for (int j = 0; j < zen_memory_pool_arr_[i]->zen_tensor_pool_size_;
               j++) {
            void *output_array =
                zen_memory_pool_arr_[i]->zen_tensor_pool_arr_[j].raw_buff;
            if (input == output_array) {
              zen_memory_pool_arr_[i]
                  ->zen_tensor_pool_arr_[j]
                  .zen_tensor_ptr_status = status;
              break;
            }
          }
        }
      }
      mtx.unlock();
    }
    // This will be enabled, when we reset pool after last zen node execution.
    if (reset) {
      ResetPoolStatus();
      zen_tensor_pool_reset_ = false;
    }
  }
};

}  // namespace amd_cpu_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_KERNELS_ZENDNN_ZEN_MEMPOOL_H_
