/*******************************************************************************
 * Modifications Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 ******************************************************************************/

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

// Standard headers.
#include <vector>

// TensorFlow plug-in headers.
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_kernel_common.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_mempool.h"
#include "tensorflow_plugin/src/amd_cpu/util/matmul_bcast.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_requires.h"
#include "tensorflow_plugin/src/amd_cpu/util/register_types.h"
#include "tensorflow_plugin/src/amd_cpu/util/zen_utils.h"

namespace amd_cpu_plugin {

// The second parameter v2_bcast is set to true if we are using V2 otherwise we
// set it to false.
template <typename Scalar, bool v2_bcast>
class ZenBatchMatMulOp : public OpKernel {
 public:
  virtual ~ZenBatchMatMulOp() {}

  explicit ZenBatchMatMulOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, InitZendnnParameters(context, &zendnn_params_));

    OP_REQUIRES_OK(context, context->GetAttr("adj_x", &adj_x_));
    OP_REQUIRES_OK(context, context->GetAttr("adj_y", &adj_y_));
  }

  void Compute(OpKernelContext *context) override {
    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-DEF: _ZenBatchMatMul (TF kernel): In Compute!");

    const Tensor &lhs = context->input(0);
    const Tensor &rhs = context->input(1);

    if (!v2_bcast) {
      // Using V1, so check to make sure lhs and rhs dimensions are correct and
      // no broadcasting is needed.
      OP_REQUIRES(context, lhs.dims() == rhs.dims(),
                  errors::InvalidArgument("lhs and rhs has different ndims: ",
                                          lhs.shape().DebugString(), " vs. ",
                                          rhs.shape().DebugString()));
      const int ndims = lhs.dims();
      OP_REQUIRES(
          context, ndims >= 2,
          errors::InvalidArgument("lhs and rhs ndims must be >= 2: ", ndims));
      for (int i = 0; i < ndims - 2; ++i) {
        OP_REQUIRES(context, lhs.dim_size(i) == rhs.dim_size(i),
                    errors::InvalidArgument(
                        "lhs.dim(", i, ") and rhs.dim(", i,
                        ") must be the same: ", lhs.shape().DebugString(),
                        " vs ", rhs.shape().DebugString()));
      }
    } else {
      OP_REQUIRES(
          context, lhs.dims() >= 2,
          errors::InvalidArgument("In[0] ndims must be >= 2: ", lhs.dims()));
      OP_REQUIRES(
          context, rhs.dims() >= 2,
          errors::InvalidArgument("In[1] ndims must be >= 2: ", rhs.dims()));
    }

    // lhs and rhs can have different dimensions.
    const int ndims_lhs = lhs.dims();
    const int ndims_rhs = rhs.dims();

    // Get broadcast information.
    MatMulBCast bcast(lhs.shape().dim_sizes(), rhs.shape().dim_sizes());
    OP_REQUIRES(
        context, bcast.IsValid(),
        errors::InvalidArgument(
            "In[0] and In[1] must have compatible batch dimensions: ",
            lhs.shape().DebugString(), " vs. ", rhs.shape().DebugString()));

    TensorShape out_shape = bcast.output_batch_shape();
    auto batch_size = bcast.output_batch_size();

    auto lhs_rows = lhs.dim_size(ndims_lhs - 2);
    auto lhs_cols = lhs.dim_size(ndims_lhs - 1);
    auto rhs_rows = rhs.dim_size(ndims_rhs - 2);
    auto rhs_cols = rhs.dim_size(ndims_rhs - 1);

    // TF-Vanilla path will override lhs_reshaped and rhs_reshaped.
    auto rhs_reshaped = rhs.template flat_inner_dims<Scalar, 3>();
    auto lhs_reshaped = lhs.template flat_inner_dims<Scalar, 3>();

    const uint64 M = lhs_reshaped.dimension(adj_x_ ? 2 : 1);
    const uint64 K = lhs_reshaped.dimension(adj_x_ ? 1 : 2);
    const uint64 N = rhs_reshaped.dimension(adj_y_ ? 1 : 2);

    // TODO(plugin): Add Switch for TF-vanilla and TF-zendnn when M and N <= 64
    // TF-Vanilla path(Eigen implementation) is optimal for batched GEMM
    // execution ZenDNN kernel works well beyond above M and N values (Currently
    // flow is going for ZenDNN implementation only).

    if (adj_x_) {
      std::swap(lhs_rows, lhs_cols);
    }
    if (adj_y_) {
      std::swap(rhs_rows, rhs_cols);
    }
    OP_REQUIRES(context, lhs_cols == rhs_rows,
                errors::InvalidArgument(
                    "lhs mismatch rhs shape: ", lhs_cols, " vs. ", rhs_rows,
                    ": ", lhs.shape().DebugString(), " ",
                    rhs.shape().DebugString(), " ", adj_x_, " ", adj_y_));

    out_shape.AddDim(lhs_rows);
    out_shape.AddDim(rhs_cols);

    // Update the output type.
    ZenTensorType out_type = ZenTensorType::kFloat;

    zendnnEnv zen_env_obj = readEnv();
    Tensor *out = nullptr;
    int zen_enable_mempool = zen_env_obj.zenEnableMemPool &&
                             !zendnn_params_.is_eager &&
                             (context->expected_output_dtype(0) == DT_FLOAT);
    ZenMemoryPool<float> *zen_pool_buffer = NULL;

    // ZenMemPool optimization reuse o/p tensors from the pool. By default its
    // enabled, export ZENDNN_ENABLE_MEMPOOL=0 will disable memory pool
    // optimization.
    // Cases where tensors in pool are not free or requested size is more than
    // available tensor size in Pool, control will fall back to default way of
    // allocation i.e. with allocate_output(..).
    if (zen_enable_mempool) {
      unsigned int thread_id = GetZenTFthreadId(std::this_thread::get_id());
      zen_pool_buffer = ZenMemoryPool<float>::GetZenMemPool(thread_id);
      if (zen_pool_buffer) {
        int status = zen_pool_buffer->AcquireZenPoolTensor(
            context, &out, out_shape, zendnn_params_.out_links,
            zendnn_params_.reset, out_type);
        if (status) {
          zen_enable_mempool = false;
        }
      } else {
        zen_enable_mempool = false;
      }
    }
    if (!zen_enable_mempool) {
      OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));
    }

    if (out->NumElements() == 0) {
      return;
    }
    if (lhs.NumElements() == 0 || rhs.NumElements() == 0) {
      zendnnInfo(ZENDNN_FWKLOG,
                 "ZEN-OP-DEF: INFO : One of the inputs has Zero elements, "
                 "Returning the control!");
      return;
    }

    auto out_reshaped = out->template flat_inner_dims<Scalar, 3>();

    std::vector<int> m_array(batch_size, M);
    std::vector<int> n_array(batch_size, N);
    std::vector<int> k_array(batch_size, K);
    std::vector<int> lda_array(batch_size, adj_x_ ? M : K);
    std::vector<int> ldb_array(batch_size, adj_y_ ? K : N);
    std::vector<int> ldc_array(batch_size, N);
    std::vector<float> alpha_array(batch_size, 1.0);
    std::vector<float> beta_array(batch_size, 0.0);
    std::vector<int> group_size(1, batch_size);
    std::vector<const Scalar *> a_array;
    std::vector<const Scalar *> b_array;
    std::vector<Scalar *> c_array;
    a_array.reserve(batch_size);
    b_array.reserve(batch_size);
    c_array.reserve(batch_size);

    if (!bcast.IsBroadcastingRequired()) {
      for (int64 i = 0; i < batch_size; i++) {
        a_array.push_back(&lhs_reshaped(i, 0, 0));
        b_array.push_back(&rhs_reshaped(i, 0, 0));
        c_array.push_back(&out_reshaped(i, 0, 0));
      }
    } else {
      // Broadcasting is needed, so get the mapping from flattened output
      // batch indices to x's and y's flattened batch indices.
      const std::vector<int64> &a_batch_indices = bcast.x_batch_indices();
      const std::vector<int64> &b_batch_indices = bcast.y_batch_indices();

      for (int64 i = 0; i < batch_size; i++) {
        a_array.push_back(&lhs_reshaped(a_batch_indices[i], 0, 0));
        b_array.push_back(&rhs_reshaped(b_batch_indices[i], 0, 0));
        c_array.push_back(&out_reshaped(i, 0, 0));
      }
    }

    bool cblasRowMajor = 1;
    zenBatchMatMul(cblasRowMajor, adj_x_, adj_y_, &m_array[0], &n_array[0],
                   &k_array[0], &alpha_array[0], &a_array[0], &lda_array[0],
                   &b_array[0], &ldb_array[0], &beta_array[0], &c_array[0],
                   &ldc_array[0], 1, &group_size[0]);

    // If ZenMemPool Optimization is enabled(default), update the state of
    // memory pool based on input_array address.
    if (zen_env_obj.zenEnableMemPool && !zendnn_params_.is_eager &&
        zen_pool_buffer) {
      zen_pool_buffer->ZenMemPoolFree(
          context, static_cast<void *>(const_cast<float *>(a_array[0])));
      zen_pool_buffer->ZenMemPoolFree(
          context, static_cast<void *>(const_cast<float *>(b_array[0])));
    }

    zendnnInfo(
        ZENDNN_FWKLOG,
        "ZEN-OP-DEF: _ZenBatchMatMul (TF kernel): Compute Is Successful!");
  }

 private:
  bool adj_x_, adj_y_;
  ZendnnParameters zendnn_params_;
};

#define REGISTER_BATCH_MATMUL_KERNELS(TYPE)                                   \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_ZenBatchMatMul").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"),   \
      ZenBatchMatMulOp<TYPE, false>);                                         \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_ZenBatchMatMulV2").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      ZenBatchMatMulOp<TYPE, true>);

TF_CALL_float(REGISTER_BATCH_MATMUL_KERNELS)

#undef REGISTER_BATCH_MATMUL_KERNELS

}  // namespace amd_cpu_plugin
