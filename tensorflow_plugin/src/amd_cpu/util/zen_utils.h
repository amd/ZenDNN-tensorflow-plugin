/*******************************************************************************
 * Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 *******************************************************************************/

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_ZEN_UTILS_H_
#define TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_ZEN_UTILS_H_

#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow_plugin/src/amd_cpu/util/env_var.h"

// NOTE: ZenDNNL header not included here. Files that need ZenDNNL should
// include "zendnnl.hpp" directly and use "using namespace zendnnl;" locally.
//
// IMPORTANT: Old ZenDNN utility classes (ZenExecutor, ZenPrimitive, etc.) have
// been REMOVED. Only MatMul uses ZenDNNL now. Other ops are disabled and will
// use vanilla TensorFlow implementations.

namespace amd_cpu_plugin {

// Old ZenDNN utility classes removed - MatMul uses ZenDNNL directly

// LRUCache is a class which implements LRU (Least Recently Used) cache.
// The implementation is taken from
//    tensorflow/core/util/mkl_util.h
//
// The LRU list maintains objects in chronological order based on creation
// time, with the least recently accessed object at the tail of LRU list, while
// the most recently accessed object at the head of LRU list.
//
// This class is used to maintain an upper bound on the total number of cached
// items. When the cache reaches its capacity, the LRU item will be removed and
// replaced by a new one from SetOp call.
//
template <typename T>
class LRUCache {
 public:
  explicit LRUCache(size_t capacity) {
    capacity_ = capacity;
    Clear();
  }

  T *GetOp(const std::string &key) {
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      return nullptr;
    }

    // Move to the front of LRU list as the most recently accessed.
    lru_list_.erase(it->second.lru_iterator);
    lru_list_.push_front(it->first);
    it->second.lru_iterator = lru_list_.begin();
    return it->second.op;
  }

  void SetOp(const std::string &key, T *op) {
    if (lru_list_.size() >= capacity_) {
      Delete();
    }

    // Insert an entry to the front of the LRU list.
    lru_list_.push_front(key);
    Entry entry(op, lru_list_.begin());
    cache_.emplace(std::make_pair(key, std::move(entry)));
  }

  void Clear() {
    if (lru_list_.empty()) {
      return;
    }

    // Clean up the cache.
    cache_.clear();
    lru_list_.clear();
  }

 private:
  struct Entry {
    // The entry's value.
    T *op;

    // A list iterator pointing to the entry's position in the LRU list.
    std::list<std::string>::iterator lru_iterator;

    // Constructor.
    Entry(T *op, std::list<std::string>::iterator it) {
      this->op = op;
      this->lru_iterator = it;
    }

    // Move constructor.
    Entry(Entry &&source) noexcept
        : lru_iterator(std::move(source.lru_iterator)) {
      op = std::move(source.op);
      source.op = std::forward<T *>(nullptr);
    }

    // Destructor.
    ~Entry() {
      if (op != nullptr) {
        delete op;
      }
    }
  };

  // Remove the least recently accessed entry from LRU list, which is the tail
  // of lru_list_. Update cache_ correspondingly.
  bool Delete() {
    if (lru_list_.empty()) {
      return false;
    }
    std::string key = lru_list_.back();
    lru_list_.pop_back();
    cache_.erase(key);
    return true;
  }

  // Cache capacity.
  size_t capacity_;

  // The cache, a map from string key to a LRU entry.
  std::unordered_map<std::string, Entry> cache_;

  // The LRU list of entries.
  // The front of the list contains the key of the most recently accessed
  // entry, while the back of the list is the least recently accessed entry.
  std::list<std::string> lru_list_;
};

// ZenPrimitive, ZenPrimitiveFactory, and FactoryKeyCreator classes removed.
// These were for old ZenDNN library and are no longer used.
// MatMul now uses ZenDNNL directly without these utility classes.

}  // namespace amd_cpu_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_ZEN_UTILS_H_
