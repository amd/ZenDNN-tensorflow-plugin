/*******************************************************************************
 * Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
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
#include "lowoha_operators/matmul/lowoha_common.hpp"
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/fill_functor.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/fused_eigen_output_kernels.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_batch_matmul_util.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_kernel_common.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_zendnnl_utils.h"
#include "tensorflow_plugin/src/amd_cpu/util/errors.h"
#include "tensorflow_plugin/src/amd_cpu/util/matmul_bcast.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_kernel.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_requires.h"
#include "tensorflow_plugin/src/amd_cpu/util/register_types.h"
#include "tensorflow_plugin/src/amd_cpu/util/tensor_format.h"
#include "tensorflow_plugin/src/amd_cpu/util/zen_utils.h"
#include "zendnnl.hpp"

namespace amd_cpu_plugin {

typedef Eigen::ThreadPoolDevice CPUDevice;

// ZenDNNL BatchMatMul implementation with optional direct API support.
template <typename T>
bool TryExecuteZenDNNLBatchMatmul(
    OpKernelContext *context, const Tensor &lhs, const Tensor &rhs,
    const Tensor *bias, Tensor *output, bool adj_x, bool adj_y, bool v2_bcast,
    FusedComputationType fusion_type, const Tensor *mul_tensor = nullptr,
    const Tensor *add_tensor = nullptr, bool use_direct_api = true) {
  // Set API name for consistent logging
  const char *api_name =
      use_direct_api ? "_ZenBatchMatMul(Direct)" : "_ZenBatchMatMul";

  // Validation: support 3D tensors and 4D tensors
  const auto ndims_lhs = lhs.shape().dims();
  const auto ndims_rhs = rhs.shape().dims();

  // TODO: Add support for 2D broadcasting (e.g., [B,M,K] x [K,N] or [M,K] x
  // [B1,B2,K,N])
  //       by expanding 2D inputs to 3D with batch=1 before computation

  // Handle 4D inputs by collapsing batch dimensions to 3D
  // This allows us to support 4D BatchMatMul with any batch sizes by:
  // 1. Reshaping 4D [B1, B2, M, K] x [B1, B2, K, N] to 3D [B1*B2, M, K] x
  // [B1*B2, K, N]
  // 2. Running 3D BatchMatMul
  // 3. Reshaping result back to 4D [B1, B2, M, N]
  bool need_reshape = false;
  Tensor lhs_reshaped_3d, rhs_reshaped_3d;
  const Tensor *lhs_to_use = &lhs;
  const Tensor *rhs_to_use = &rhs;
  int64 original_batch_dim0 = 1;
  int64 original_batch_dim1 = 1;

  // TensorFlow BatchMatMulV2 Broadcasting Rules (dimension-by-dimension from
  // RIGHT to LEFT):
  //   - Each dimension pair must: MATCH EXACTLY or ONE must be 1
  // Invalid: [2, 3, M, K] x [3, 3, K, N]  → B1: 2≠3, neither is 1
  // Invalid: [2, 3, M, K] x [6, K, N]  → [2,3] vs [1,6], B2: 3≠6, neither is 1
  // Invalid: [6, M, K] x [3, K, N]  → B: 6≠3, neither is 1

  // Helper lambda to create and validate reshaped 3D tensor
  // Note: Uses original tensor dimensions (before transpose via adj_x/adj_y)
  // Transpose is applied later in computation, not during reshaping
  auto create_reshaped_3d = [&](const Tensor &input, int64 batch,
                                int input_ndims, int dim_offset,
                                Tensor &output) -> bool {
    TensorShape shape;
    shape.AddDim(batch);
    shape.AddDim(input.dim_size(input_ndims - 2 + dim_offset));
    shape.AddDim(input.dim_size(input_ndims - 1 + dim_offset));

    if (!output.CopyFrom(input, shape)) {
      LogZenDNNLInfo(api_name, "Failed to create reshaped tensor view");
      return false;
    }
    return true;
  };

  // Check if we need to handle 4D inputs or mixed dimensions
  if ((ndims_lhs == 4 || ndims_rhs == 4) &&
      (ndims_lhs >= 3 && ndims_rhs >= 3)) {
    // Mixed dimension check: reject V1 for 4D x 3D or 3D x 4D
    if ((ndims_lhs != ndims_rhs) && !v2_bcast) {
      LogZenDNNLInfo(api_name, "Mixed dimension broadcasting requires V2");
      return false;
    }

    // Get batch dimensions (treat 3D as 4D with implicit [1, B])
    int64 batch_dim0_lhs = (ndims_lhs == 4) ? lhs.dim_size(0) : 1;
    int64 batch_dim1_lhs = (ndims_lhs == 4) ? lhs.dim_size(1) : lhs.dim_size(0);
    int64 batch_dim0_rhs = (ndims_rhs == 4) ? rhs.dim_size(0) : 1;
    int64 batch_dim1_rhs = (ndims_rhs == 4) ? rhs.dim_size(1) : rhs.dim_size(0);

    // Validate batch compatibility
    if ((batch_dim0_lhs != batch_dim0_rhs && batch_dim0_lhs != 1 &&
         batch_dim0_rhs != 1) ||
        (batch_dim1_lhs != batch_dim1_rhs && batch_dim1_lhs != 1 &&
         batch_dim1_rhs != 1)) {
      LogZenDNNLInfo(api_name, ("Incompatible batch dimensions: LHS=[" +
                                std::to_string(batch_dim0_lhs) + "," +
                                std::to_string(batch_dim1_lhs) + "], RHS=[" +
                                std::to_string(batch_dim0_rhs) + "," +
                                std::to_string(batch_dim1_rhs) + "]")
                                   .c_str());
      return false;
    }

    // Calculate effective batch and reshape to 3D
    // Note: TensorFlow tensors are row-major (C-order) by default
    // Verify layout before proceeding with reshape
    original_batch_dim0 = std::max(batch_dim0_lhs, batch_dim0_rhs);
    original_batch_dim1 = std::max(batch_dim1_lhs, batch_dim1_rhs);

    int64 collapsed_batch_lhs = batch_dim0_lhs * batch_dim1_lhs;
    int64 collapsed_batch_rhs = batch_dim0_rhs * batch_dim1_rhs;

    // Reshape both inputs to 3D using their respective collapsed batches
    need_reshape = true;

    if (!create_reshaped_3d(lhs, collapsed_batch_lhs, ndims_lhs, 0,
                            lhs_reshaped_3d) ||
        !create_reshaped_3d(rhs, collapsed_batch_rhs, ndims_rhs, 0,
                            rhs_reshaped_3d)) {
      LogZenDNNLInfo(api_name, "Failed to create reshaped tensor view");
      return false;
    }

    // Validate reshaped tensors are not empty
    if (lhs_reshaped_3d.NumElements() == 0 ||
        rhs_reshaped_3d.NumElements() == 0) {
      LogZenDNNLInfo(api_name, "Reshaped tensor is empty");
      return false;
    }

    lhs_to_use = &lhs_reshaped_3d;
    rhs_to_use = &rhs_reshaped_3d;

    // Note: Add tensor doesn't need reshaping - ZenDNN library handles
    // batch broadcasting for 2D post-ops internally
  } else if (ndims_lhs != 3 || ndims_rhs != 3) {
    LogZenDNNLInfo(api_name, "Only 3D and 4D inputs supported");
    return false;
  }

  // Common setup: Determine data type
  bool is_float = std::is_same<T, float>::value;

  // Calculate M, N, K dimensions from input tensors
  // For batch matmul: lhs is [batch, M, K] and rhs is [batch, K, N] (before
  // transpose) After adj_x/adj_y, the actual dimensions are adjusted
  // Use squeezed tensors if we're processing 4D inputs

  const auto effective_ndims_lhs = lhs_to_use->shape().dims();
  const auto effective_ndims_rhs = rhs_to_use->shape().dims();

  auto lhs_rows = lhs_to_use->dim_size(effective_ndims_lhs - 2);
  auto lhs_cols = lhs_to_use->dim_size(effective_ndims_lhs - 1);
  auto rhs_rows = rhs_to_use->dim_size(effective_ndims_rhs - 2);
  auto rhs_cols = rhs_to_use->dim_size(effective_ndims_rhs - 1);

  // Calculate logical dimensions after transpose
  // M = number of rows in result = rows of lhs after transpose
  // N = number of cols in result = cols of rhs after transpose
  // K = shared inner dimension
  uint64_t M = adj_x ? lhs_cols : lhs_rows;
  uint64_t K = adj_x ? lhs_rows : lhs_cols;
  uint64_t N = adj_y ? rhs_rows : rhs_cols;

  // Validate K dimensions match
  uint64_t K_rhs = adj_y ? rhs_cols : rhs_rows;
  if (K != K_rhs) {
    LogZenDNNLInfo(api_name,
                   ("K dimension mismatch: LHS K=" + std::to_string(K) +
                    ", RHS K=" + std::to_string(K_rhs))
                       .c_str());
    return false;
  }

  // Get batch dimensions
  int batch_lhs = static_cast<int>(lhs_to_use->dim_size(0));
  int batch_rhs = static_cast<int>(rhs_to_use->dim_size(0));

  // Create a reshaped view of the output if inputs were reshaped
  Tensor output_reshaped_3d;
  Tensor *output_to_use = output;

  if (need_reshape) {
    // Output is 4D [B1, B2, M, N], we need 3D [B1*B2, M, N] for computation
    TensorShape output_reshaped_shape;
    int64 collapsed_batch = original_batch_dim0 * original_batch_dim1;
    output_reshaped_shape.AddDim(collapsed_batch);

    // Get M, N from output shape (last 2 dimensions)
    auto ndims_out = output->shape().dims();
    output_reshaped_shape.AddDim(output->dim_size(ndims_out - 2));  // M
    output_reshaped_shape.AddDim(output->dim_size(ndims_out - 1));  // N

    // Create a reshaped view of the output tensor (no data copy)
    if (!output_reshaped_3d.CopyFrom(*output, output_reshaped_shape)) {
      LogZenDNNLInfo(api_name,
                     "Failed to create reshaped view of output tensor");
      return false;
    }
    output_to_use = &output_reshaped_3d;
  }

  // Compute expected output shape for broadcasting validation
  auto lhs_shape = lhs_to_use->shape();
  auto rhs_shape = rhs_to_use->shape();
  auto out_shape = output_to_use->shape();
  const auto ndims_out = out_shape.dims();

  MatMulBCast bcast(lhs_shape.dim_sizes(), rhs_shape.dim_sizes());
  if (!bcast.IsValid()) {
    LogZenDNNLInfo(api_name, "Broadcast validation failed for input shapes");
    return false;
  }

  TensorShape computed_out_shape = bcast.output_batch_shape();
  computed_out_shape.AddDim(M);
  computed_out_shape.AddDim(N);

  const auto ndims_computed_out = computed_out_shape.dims();

  // Extract dimensions for broadcasting
  auto lhs_dims = ExtractDimsFromTFShape(lhs_shape);
  auto rhs_dims = ExtractDimsFromTFShape(rhs_shape);

  // Check if broadcasting is needed
  bool lhs_flag = false, rhs_flag = false;
  const auto ndims_lhs_effective = lhs_shape.dims();
  const auto ndims_rhs_effective = rhs_shape.dims();

  if (ndims_lhs_effective < ndims_computed_out) {
    ExpandInputDimsToOutputShape(lhs_shape, computed_out_shape, &lhs_dims);
    lhs_flag = true;
  }
  if (ndims_rhs_effective < ndims_computed_out) {
    ExpandInputDimsToOutputShape(rhs_shape, computed_out_shape, &rhs_dims);
    rhs_flag = true;
  }

  // Use tensor aliasing to avoid data copies for broadcasting
  // When broadcasting adds leading dimensions of size 1, we can create views
  // of the original data with expanded shape (no data copy needed)
  Tensor lhs_reshaped_tensor;
  Tensor rhs_reshaped_tensor;

  if (rhs_flag) {
    TensorShape rhs_reshaped_shape(rhs_dims);
    // Create a view with expanded dimensions (no data copy)
    if (!rhs_reshaped_tensor.CopyFrom(*rhs_to_use, rhs_reshaped_shape)) {
      LogZenDNNLInfo(
          api_name,
          "Failed to create reshaped view of RHS tensor for broadcasting");
      return false;
    }
  }

  if (lhs_flag) {
    TensorShape lhs_reshaped_shape(lhs_dims);
    // Create a view with expanded dimensions (no data copy)
    if (!lhs_reshaped_tensor.CopyFrom(*lhs_to_use, lhs_reshaped_shape)) {
      LogZenDNNLInfo(
          api_name,
          "Failed to create reshaped view of LHS tensor for broadcasting");
      return false;
    }
  }

  // Use reshaped tensors if broadcasting was performed
  const Tensor &lhs_for_compute = lhs_flag ? lhs_reshaped_tensor : *lhs_to_use;
  const Tensor &rhs_for_compute = rhs_flag ? rhs_reshaped_tensor : *rhs_to_use;

  // Update batch dimensions from (potentially reshaped) tensors
  batch_lhs = static_cast<int>(lhs_for_compute.dim_size(0));
  batch_rhs = static_cast<int>(rhs_for_compute.dim_size(0));

  // Try direct API first if enabled and applicable (3D only)
  // Note: effective dimensions are 3D even if original inputs were 4D with
  // batch=1
  if (use_direct_api && effective_ndims_lhs == 3 && effective_ndims_rhs == 3) {
    try {
      using namespace zendnnl::lowoha::matmul;
      using namespace zendnnl::memory;
      using namespace zendnnl::ops;

      // Setup batch parameters
      matmul_batch_params_t batch_params;
      batch_params.Batch_A = batch_lhs;
      batch_params.Batch_B = batch_rhs;

      // Calculate strides from original tensor layout (before MatMulBCast
      // expansion) If original batch=1, data is not duplicated, stride=0 for
      // broadcasting If original batch>1, stride=M*K (or K*N) for actual batch
      // layout
      int64 original_batch_lhs = lhs_to_use->dim_size(0);
      int64 original_batch_rhs = rhs_to_use->dim_size(0);

      batch_params.batch_stride_src = (original_batch_lhs == 1) ? 0 : M * K;
      batch_params.batch_stride_wei = (original_batch_rhs == 1) ? 0 : K * N;
      batch_params.batch_stride_dst = M * N;

      // Setup lowoha parameters
      matmul_params params;
      data_type_t dt = is_float ? data_type_t::f32 : data_type_t::bf16;
      params.dtypes.src = dt;
      params.dtypes.wei = dt;
      params.dtypes.dst = dt;
      if (bias) {
        params.dtypes.bias = dt;
      }
      params.dtypes.compute = dt;

      params.mem_format_a = 'n';
      params.mem_format_b = 'n';
      float alpha = 1.0f;

      // ToDO: Generalize the below code and remove duplicate code.
      // Setup post-ops based on fusion type
      switch (fusion_type) {
        case FusedComputationType::kBiasAddWithRelu: {
          matmul_post_op relu_postop;
          relu_postop.po_type = post_op_type_t::relu;
          params.postop_.push_back(relu_postop);
          break;
        }
        case FusedComputationType::kBinaryMul: {
          if (mul_tensor) {
            // Remapper only creates fusion for scalar mul
            if (mul_tensor->NumElements() != 1) {
              LogZenDNNLInfo(api_name, "Binary mul tensor is non-scalar");
              return false;
            }
            alpha = static_cast<float>(mul_tensor->flat<T>()(0));
          }
          break;
        }
        case FusedComputationType::kBinaryMulAdd: {
          if (mul_tensor && add_tensor) {
            // Validate tensors per remapper constraints:
            // 1. multiplicand must be scalar (NumCoefficients == 1)
            // 2. addend must be 4D with shape [1, 1, M, N]
            if (mul_tensor->NumElements() != 1) {
              LogZenDNNLInfo(api_name, "Binary mul tensor must be scalar");
              return false;
            }

            // Remapper enforces: addend rank == 4 && dim(0) == 1 && dim(1) == 1
            if (add_tensor->dims() != 4 || add_tensor->dim_size(0) != 1 ||
                add_tensor->dim_size(1) != 1) {
              LogZenDNNLInfo(api_name,
                             "Binary add tensor must have shape [1, 1, M, N], "
                             "actual shape");
              return false;
            }

            // Scalar multiplication via alpha
            alpha = static_cast<float>(mul_tensor->flat<T>()(0));

            // Binary add post-op - extract last 2 dims from [1, 1, M, N]
            matmul_post_op add_postop;
            add_postop.po_type = post_op_type_t::binary_add;
            add_postop.dtype = dt;
            add_postop.buff = const_cast<T *>(add_tensor->flat<T>().data());
            add_postop.dims = {add_tensor->dim_size(2),
                               add_tensor->dim_size(3)};

            params.postop_.push_back(add_postop);
          }
          break;
        }
        default:
          break;
      }

      // Get data pointers - use reshaped tensors if broadcasting was performed
      const T *lhs_data = lhs_for_compute.flat<T>().data();
      const T *rhs_data = rhs_for_compute.flat<T>().data();
      T *output_data = output_to_use->flat<T>().data();
      const T *bias_data = (bias) ? bias->flat<T>().data() : nullptr;

      // Leading dimensions
      int lda = adj_x ? M : K;
      int ldb = adj_y ? K : N;
      int ldc = N;

      status_t status = matmul_direct('r', adj_x, adj_y, M, N, K, alpha,
                                      static_cast<const void *>(lhs_data), lda,
                                      static_cast<const void *>(rhs_data), ldb,
                                      static_cast<const void *>(bias_data),
                                      0.0f, static_cast<void *>(output_data),
                                      ldc, false, batch_params, params);

      if (status == status_t::success) {
        LogZenDNNLSuccess(api_name);
        return true;
      } else {
        LogZenDNNLInfo(api_name, ("Execution failed with status " +
                                  std::to_string(static_cast<int>(status)))
                                     .c_str());
        return false;
      }

    } catch (const std::exception &e) {
      LogZenDNNLFallback(api_name,
                         ("Exception: " + std::string(e.what())).c_str());
      return false;
    }
  }

  // Operator API implementation
  try {
    using namespace zendnnl;
    using namespace zendnnl::memory;
    using namespace zendnnl::common;

    // Determine data type
    data_type_t dt = is_float ? data_type_t::f32 : data_type_t::bf16;

    // Get pointers to TensorFlow tensor data
    T *lhs_data = const_cast<T *>(lhs_for_compute.flat<T>().data());
    T *rhs_data = const_cast<T *>(rhs_for_compute.flat<T>().data());
    T *output_data = output_to_use->flat<T>().data();

    // Calculate buffer sizes
    uint64_t lhs_buffer_size = lhs_for_compute.NumElements() * sizeof(T);
    uint64_t rhs_buffer_size = rhs_for_compute.NumElements() * sizeof(T);
    uint64_t out_buffer_size = output_to_use->NumElements() * sizeof(T);

    // Create tensors with logical dimensions (batch + M,K,N)
    tensor_t input_lhs, input_rhs, output_tensor;

    // For 3D BatchMatMul: batch dim + logical M,K,N
    std::vector<uint64_t> lhs_tensor_dims = {static_cast<uint64_t>(batch_lhs),
                                             M, K};
    std::vector<uint64_t> rhs_tensor_dims = {static_cast<uint64_t>(batch_rhs),
                                             K, N};
    std::vector<uint64_t> out_tensor_dims = {static_cast<uint64_t>(batch_lhs),
                                             M, N};

    // Setup lhs tensor - borrow memory from TF tensor.
    // set_size() receives logical dims; order string indicates physical memory
    // layout. For 3D: use "abc" normally, "acb" if adj_x=true (transposed).
    std::string lhs_order = adj_x ? "acb" : "abc";

    input_lhs.set_size(lhs_tensor_dims)
        .set_data_type(dt)
        .set_order(lhs_order)
        .set_storage(static_cast<void *>(lhs_data), lhs_buffer_size)
        .create();

    // Setup rhs tensor - borrow memory from TF tensor.
    // For 3D: use "abc" normally, "acb" if adj_y=true (transposed).
    std::string rhs_order = adj_y ? "acb" : "abc";

    input_rhs.set_size(rhs_tensor_dims)
        .set_data_type(dt)
        .set_order(rhs_order)
        .set_storage(static_cast<void *>(rhs_data), rhs_buffer_size)
        .set_name("weights")
        .create();

    // Setup output tensor - borrow memory from TF tensor.
    // For 3D: always use "abc" (output is never transposed).
    std::string out_order = "abc";

    output_tensor.set_size(out_tensor_dims)
        .set_data_type(dt)
        .set_order(out_order)
        .set_storage(static_cast<void *>(output_data), out_buffer_size)
        .create();

    // Create matmul context.
    using namespace zendnnl::ops;
    matmul_context_t matmul_context;
    matmul_context.set_param("weights", input_rhs);

    // Handle bias for BiasAdd fusion types.
    tensor_t bias_tensor;
    if (bias && fusion_type != FusedComputationType::kUndefined &&
        fusion_type != FusedComputationType::kBinaryMul &&
        fusion_type != FusedComputationType::kBinaryMulAdd) {
      T *bias_data = const_cast<T *>(bias->flat<T>().data());
      uint64_t bias_buffer_size = bias->NumElements() * sizeof(T);

      // Bias dimensions for 3D batch matmul (N is the last dimension of
      // output).
      std::vector<uint64_t> bias_dims = {1, 1, N};

      bias_tensor.set_size(bias_dims)
          .set_data_type(dt)
          .set_storage(static_cast<void *>(bias_data), bias_buffer_size)
          .set_name("bias")
          .create();

      matmul_context.set_param("bias", bias_tensor);
    }

    // Declare post-op tensors in outer scope for later use when setting
    // operator inputs.
    tensor_t add_post_op_tensor, mul_post_op_tensor;
    float alpha = 1.0f;
    // Add post-ops based on fusion type.
    switch (fusion_type) {
      case FusedComputationType::kBiasAddWithRelu: {
        post_op_t relu_post_op(post_op_type_t::relu);
        matmul_context.set_post_op(relu_post_op);
        break;
      }
      case FusedComputationType::kBinaryMul: {
        // Binary multiply post-op (Remapper enforces: scalar multiplicand)
        if (mul_tensor) {
          if (mul_tensor->NumElements() != 1) {
            LogZenDNNLInfo(api_name, "Binary mul tensor must be scalar");
            return false;
          }
          alpha = static_cast<float>(mul_tensor->flat<T>()(0));
          matmul_context.set_alpha(alpha);
        } else {
          LogZenDNNLInfo(api_name,
                         "Binary mul requested but no mul tensor provided");
          return false;
        }
        break;
      }
      case FusedComputationType::kBinaryMulAdd: {
        // Binary multiply + add post-ops.
        if (mul_tensor && add_tensor) {
          // Validate tensors per remapper constraints:
          // 1. multiplicand must be scalar (NumCoefficients == 1)
          // 2. addend must be 4D with shape [1, 1, M, N]
          if (mul_tensor->NumElements() != 1) {
            LogZenDNNLInfo(api_name, "Binary mul tensor must be scalar");
            return false;
          }

          // Remapper enforces: addend rank == 4 && dim(0) == 1 && dim(1) == 1
          if (add_tensor->dims() != 4 || add_tensor->dim_size(0) != 1 ||
              add_tensor->dim_size(1) != 1) {
            LogZenDNNLInfo(api_name,
                           "Binary add tensor must have shape [1, 1, M, N], "
                           "actual shape");
            return false;
          }

          // Scalar multiplication via alpha
          alpha = static_cast<float>(mul_tensor->flat<T>()(0));
          matmul_context.set_alpha(alpha);

          // Handle add post-op - extract last 2 dims from [1, 1, M, N]
          T *add_data = const_cast<T *>(add_tensor->flat<T>().data());
          uint64_t add_buffer_size = add_tensor->NumElements() * sizeof(T);

          std::vector<uint64_t> add_dims = {
              static_cast<uint64_t>(add_tensor->dim_size(2)),
              static_cast<uint64_t>(add_tensor->dim_size(3))};

          add_post_op_tensor.set_size(add_dims)
              .set_data_type(dt)
              .set_storage(static_cast<void *>(add_data), add_buffer_size)
              .set_name("binary_add_tensor_1")
              .create();

          binary_add_params_t add_params{1.0f, "binary_add_tensor_1"};
          post_op_t add_post_op(add_params);
          matmul_context.set_post_op(add_post_op);
          matmul_context.set_param("binary_add_tensor_1", add_post_op_tensor);
        } else {
          LogZenDNNLInfo(
              api_name,
              "Binary mul+add requested but mul/add tensors not provided");
          return false;
        }
        break;
      }
      default:
        // No post-op or unsupported fusion type.
        break;
    }

    matmul_context.create();

    // Create matmul operator.
    matmul_operator_t matmul_operator;
    matmul_operator.set_name("tf_zendnnl_batch_matmul")
        .set_context(matmul_context)
        .create();

    // Set input/output tensor names.
    input_lhs.set_name("matmul_input");
    output_tensor.set_name("matmul_output");

    matmul_operator.set_input("matmul_input", input_lhs);
    matmul_operator.set_output("matmul_output", output_tensor);

    // Set post-op tensors as inputs by querying context for post-ops.
    int post_op_count = matmul_context.get_post_op_count();
    for (int i = 0; i < post_op_count; ++i) {
      auto post_op = matmul_context.get_post_op(i);
      if (post_op.type == post_op_type_t::binary_mul) {
        matmul_operator.set_input(
            matmul_context.get_post_op(i).binary_mul_params.tensor_name,
            mul_post_op_tensor);
      } else if (post_op.type == post_op_type_t::binary_add) {
        matmul_operator.set_input(
            matmul_context.get_post_op(i).binary_add_params.tensor_name,
            add_post_op_tensor);
      }
    }

    status_t status = matmul_operator.execute();

    if (status != status_t::success) {
      LogZenDNNLInfo(api_name, ("Execution failed with status " +
                                std::to_string(static_cast<int>(status)))
                                   .c_str());
      return false;
    }
    LogZenDNNLSuccess(api_name);
    return true;

  } catch (const std::exception &e) {
    LogZenDNNLFallback(api_name,
                       ("Exception: " + std::string(e.what())).c_str());
    return false;
  }
}

// Explicit template instantiations for float and bfloat16.
template bool TryExecuteZenDNNLBatchMatmul<float>(
    OpKernelContext *, const Tensor &, const Tensor &, const Tensor *, Tensor *,
    bool, bool, bool, FusedComputationType, const Tensor *, const Tensor *,
    bool);
template bool TryExecuteZenDNNLBatchMatmul<Eigen::bfloat16>(
    OpKernelContext *, const Tensor &, const Tensor &, const Tensor *, Tensor *,
    bool, bool, bool, FusedComputationType, const Tensor *, const Tensor *,
    bool);

// The second parameter v2_bcast is set to true if we are using V2 otherwise we
// set it to false.
template <typename Scalar, bool v2_bcast, bool fusion_enabled>
class ZenBatchMatMulOp : public OpKernel {
 public:
  virtual ~ZenBatchMatMulOp() {}

  explicit ZenBatchMatMulOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("adj_x", &adj_x_));
    OP_REQUIRES_OK(context, context->GetAttr("adj_y", &adj_y_));

    OP_REQUIRES_OK(context,
                   context->GetAttr("is_cache_weight", &is_cache_weight_));

    std::vector<FusedComputationPattern> patterns;
    if (fusion_enabled) {
      using FCT = FusedComputationType;

      patterns = {{FCT::kBinaryMul, {"BinaryMul"}},
                  {FCT::kBinaryMulAdd, {"BinaryMul", "Add"}},
                  {FCT::kBiasAddWithRelu, {"BiasAdd", "Relu"}}};

      OP_REQUIRES_OK(context,
                     InitializeFusedComputation(context, "_ZenBatchMatMul",
                                                patterns, &fused_computation_,
                                                &fused_computation_args_));
    }
  }

  void Compute(OpKernelContext *context) override {
    const Tensor &lhs = context->input(0);
    const Tensor &rhs = context->input(1);

    if (!v2_bcast) {
      // Using V1, so check to make sure lhs and rhs dimensions are correct
      // and no broadcasting is needed.
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

    const auto ndims_lhs = lhs.dims();
    const auto ndims_rhs = rhs.dims();
    const int ndims = lhs.dims();

    // Support 3D and 4D inputs
    OP_REQUIRES(
        context, ndims == 3 || ndims == 4,
        errors::InvalidArgument(
            "BatchMatMul is supported for 3D and 4D inputs only, got ndims=",
            ndims));

    MatMulBCast bcast(lhs.shape().dim_sizes(), rhs.shape().dim_sizes());
    OP_REQUIRES(
        context, bcast.IsValid(),
        errors::InvalidArgument(
            "In[0] and In[1] must have compatible batch dimensions: ",
            lhs.shape().DebugString(), " vs. ", rhs.shape().DebugString()));

    TensorShape out_shape = bcast.output_batch_shape();

    auto lhs_rows = lhs.dim_size(ndims_lhs - 2);
    auto lhs_cols = lhs.dim_size(ndims_lhs - 1);
    auto rhs_rows = rhs.dim_size(ndims_rhs - 2);
    auto rhs_cols = rhs.dim_size(ndims_rhs - 1);

    if (adj_x_) std::swap(lhs_rows, lhs_cols);
    if (adj_y_) std::swap(rhs_rows, rhs_cols);

    out_shape.AddDim(lhs_rows);
    out_shape.AddDim(rhs_cols);

    // Allocate output tensor
    Tensor *output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    // Handle empty tensors
    if (output->NumElements() == 0) {
      return;
    }

    if (lhs.NumElements() == 0 || rhs.NumElements() == 0) {
      functor::SetZeroFunctor<CPUDevice, Scalar> f;
      f(context->eigen_cpu_device(), output->flat<Scalar>());
      return;
    }

    // Try ZenDNNL implementation if enabled
    const Tensor *bias = nullptr;
    const Tensor *mul_tensor = nullptr;
    const Tensor *add_tensor = nullptr;

    // Extract fusion tensors
    if (fusion_enabled &&
        fused_computation_ == FusedComputationType::kBiasAddWithRelu) {
      bias = &context->input(2);
    }

    if (fusion_enabled &&
        (fused_computation_ == FusedComputationType::kBinaryMul ||
         fused_computation_ == FusedComputationType::kBinaryMulAdd)) {
      mul_tensor = &context->input(2);
    }

    if (fusion_enabled &&
        fused_computation_ == FusedComputationType::kBinaryMulAdd) {
      add_tensor = &context->input(3);
    }

    // Check if ZenDNNL direct API should be used
    bool use_direct_api = IsZenDnnMatmulDirectEnabled();

    bool zendnnl_success = TryExecuteZenDNNLBatchMatmul<Scalar>(
        context, lhs, rhs, bias, output, adj_x_, adj_y_, v2_bcast,
        fused_computation_, mul_tensor, add_tensor, use_direct_api);

    OP_REQUIRES(context, zendnnl_success,
                errors::Internal("_ZenBatchMatMul execution failed"));
  }

 private:
  bool adj_x_ = false;
  bool adj_y_ = false;
  bool is_cache_weight_ = false;
  FusedComputationType fused_computation_ = FusedComputationType::kUndefined;
  FusedComputationArgs fused_computation_args_;
};
#define REGISTER_BATCH_MATMUL_KERNELS(TYPE)                                   \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_ZenBatchMatMul").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"),   \
      ZenBatchMatMulOp<TYPE, false, false>);                                  \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_ZenBatchMatMulV2").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      ZenBatchMatMulOp<TYPE, true, false>);                                   \
  REGISTER_KERNEL_BUILDER(Name("_ZenFusedBatchMatMulV2")                      \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<TYPE>("T"),                     \
                          ZenBatchMatMulOp<TYPE, true, true>);

TF_CALL_float(REGISTER_BATCH_MATMUL_KERNELS);
TF_CALL_bfloat16(REGISTER_BATCH_MATMUL_KERNELS);
#undef REGISTER_BATCH_MATMUL_KERNELS

}  // namespace amd_cpu_plugin
