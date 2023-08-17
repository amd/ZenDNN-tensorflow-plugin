/*******************************************************************************
 * Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights
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
#include "zendnn.hpp"          // NOLINT(build/include_subdir)
#include "zendnn_helper.hpp"   // NOLINT(build/include_subdir)
#include "zendnn_logging.hpp"  // NOLINT(build/include_subdir)

using zendnn::_zendnnGetLogState;
using zendnn::_zendnnLogMessage;
using zendnn::algorithm;
using zendnn::engine;
using zendnn::memory;
using zendnn::post_ops;
using zendnn::primitive;
using zendnn::prop_kind;
using zendnn::stream;
using zendnn::zenConvAlgoType;
using zendnn::ZENDNN_FWKLOG;
using zendnn::zendnnEnv;

namespace amd_cpu_plugin {

// For single engine and stream
// TODO(zendnn): Need a complete graph manager entity. This can be moved to
// within that.
class ZenExecutor {
 private:
  inline static ZenExecutor *instance = 0;
  engine eng_;
  std::vector<std::shared_ptr<stream>> engine_stream_;

  ZenExecutor() {
    engine temp_eng(engine::kind::cpu, 0);
    eng_ = temp_eng;
    std::shared_ptr<stream> temp_stream = std::make_shared<stream>(eng_);
    std::vector<std::shared_ptr<stream>> temp_vec_stream = {temp_stream};
    engine_stream_ = temp_vec_stream;
  }

 public:
  static ZenExecutor *getInstance() {
    if (!instance) {
      instance = new ZenExecutor();
    }
    return instance;
  }

  engine getEngine() { return eng_; }

  stream getStream() {
    std::shared_ptr<stream> s = engine_stream_[engine_stream_.size() - 1];
    stream res = *s;
    return res;
  }

  std::shared_ptr<stream> getStreamPtr() {
    return engine_stream_[engine_stream_.size() - 1];
  }

  void addStream() {
    std::shared_ptr<stream> temp_stream = std::make_shared<stream>(eng_);
    engine_stream_.push_back(temp_stream);
  }
};

inline void execute_primitives(
    const std::vector<primitive> &primitives, std::shared_ptr<stream> stream,
    const std::vector<std::unordered_map<int, memory>> &net_args) {
  DCHECK_EQ(primitives.size(), net_args.size());
  for (size_t i = 0; i < primitives.size(); ++i) {
    primitives.at(i).execute(*stream, net_args.at(i));
  }
}

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

class ZenPrimitive {
 public:
  virtual ~ZenPrimitive() {}
  ZenPrimitive() {
    ZenExecutor *ex = ex->getInstance();
    cpu_engine_ = ex->getEngine();
  }
  explicit ZenPrimitive(const engine &cpu_engine) { cpu_engine_ = cpu_engine; }
  unsigned char *DummyData = nullptr;
  engine cpu_engine_;
  const engine &GetEngine() { return cpu_engine_; }
};

class ZenPrimitiveFactory {
 public:
  ZenPrimitiveFactory() {}

  ~ZenPrimitiveFactory() {}

  ZenPrimitive *GetOp(const std::string &key) {
    auto &lru_cache = ZenPrimitiveFactory::GetLRUCache();
    return lru_cache.GetOp(key);
  }

  void SetOp(const std::string &key, ZenPrimitive *op) {
    auto &lru_cache = ZenPrimitiveFactory::GetLRUCache();
    lru_cache.SetOp(key, op);
  }

  // Function to decide whether HW has AVX512 or AVX2.
  static inline bool IsLegacyPlatform() {
    return (
        !tensorflow::port::TestCPUFeature(
            tensorflow::port::CPUFeature::AVX512F) &&
        !tensorflow::port::TestCPUFeature(tensorflow::port::CPUFeature::AVX2));
  }

  // Function to check whether primitive reuse optimization is disabled.
  static inline bool IsReuseOptDisabled() {
    bool is_reuse_opt_disabled = false;
    TF_CHECK_OK(ReadBoolFromEnvVar("TF_ZEN_PRIMITIVE_REUSE_DISABLE", false,
                                   &is_reuse_opt_disabled));
    return is_reuse_opt_disabled;
  }

 private:
  static inline LRUCache<ZenPrimitive> &GetLRUCache() {
    static const int kCapacity = 1024;  // cache capacity
    static thread_local LRUCache<ZenPrimitive> lru_cache_(kCapacity);
    return lru_cache_;
  }
};

// Utility class for creating keys of Zen primitive pool.
// The implementation is taken from : tensorflow/core/util/mkl_util.h.
class FactoryKeyCreator {
 public:
  FactoryKeyCreator() { key_.reserve(kMaxKeyLength); }

  ~FactoryKeyCreator() {}

  void AddAsKey(const std::string &str) { Append(str); }

  void AddAsKey(const memory::dims &dims) {
    for (unsigned int i = 0; i < dims.size(); i++) {
      AddAsKey<int>(dims[i]);
    }
  }

  template <typename T>
  void AddAsKey(const T data) {
    auto buffer = reinterpret_cast<const char *>(&data);
    Append(StringPiece(buffer, sizeof(T)));
  }

  std::string GetKey() { return key_; }

 private:
  std::string key_;
  const char kDelimiter = 'x';
  const int kMaxKeyLength = 256;
  void Append(StringPiece s) {
    key_.append(std::string(s));
    key_.append(1, kDelimiter);
  }
};

}  // namespace amd_cpu_plugin

#endif  // TENSORFLOW_PLUGIN_SRC_AMD_CPU_UTIL_ZEN_UTILS_H_
