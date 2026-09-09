/*******************************************************************************
 * Copyright (c) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 *******************************************************************************/

#include <string>
#include <vector>

// TensorFlow plug-in headers.
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/fill_functor.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/fused_eigen_output_kernels.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_einsum_patterns.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_kernel_common.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_zendnnl_utils.h"
#include "tensorflow_plugin/src/amd_cpu/util/errors.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_kernel.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_requires.h"
#include "tensorflow_plugin/src/amd_cpu/util/register_types.h"
#include "tensorflow_plugin/src/amd_cpu/util/tensor_format.h"
#include "tensorflow_plugin/src/amd_cpu/util/zen_utils.h"
#include "zendnnl.hpp"
// ZenDNNL logging support
#include "common/zendnnl_global.hpp"

namespace amd_cpu_plugin {

typedef Eigen::ThreadPoolDevice CPUDevice;

// Execute einsum as 2D matmul using ZenDNN's matmul_direct API.
// This handles the "ij,jk->ik" pattern.
template <typename T>
bool TryExecuteZenDNNLEinsumAsMatMul(OpKernelContext* context, const Tensor& a,
                                     const Tensor& b, Tensor* output,
                                     bool transpose_a, bool transpose_b) {
  try {
    using namespace zendnnl::common;
    using namespace zendnnl::memory;
    using namespace zendnnl::ops;
    using namespace zendnnl::lowoha::matmul;

    const bool use_direct_api = IsZenDnnMatmulDirectEnabled();
    const char* api_name = use_direct_api ? "_ZenEinsum(Direct)" : "_ZenEinsum";

    // Get tensor dimensions.
    auto a_shape = a.shape();
    auto b_shape = b.shape();

    // Validate 2D inputs for matmul pattern.
    if (a_shape.dims() != 2 || b_shape.dims() != 2) {
      LogZenDNNLInfo(api_name, "Einsum matmul pattern requires 2D inputs");
      return false;
    }

    // Calculate dimensions.
    uint64_t m, k_a, k_b, n;
    if (transpose_a) {
      m = static_cast<uint64_t>(a_shape.dim_size(1));
      k_a = static_cast<uint64_t>(a_shape.dim_size(0));
    } else {
      m = static_cast<uint64_t>(a_shape.dim_size(0));
      k_a = static_cast<uint64_t>(a_shape.dim_size(1));
    }

    if (transpose_b) {
      k_b = static_cast<uint64_t>(b_shape.dim_size(1));
      n = static_cast<uint64_t>(b_shape.dim_size(0));
    } else {
      k_b = static_cast<uint64_t>(b_shape.dim_size(0));
      n = static_cast<uint64_t>(b_shape.dim_size(1));
    }

    // Validate inner dimensions match.
    if (k_a != k_b) {
      LogZenDNNLInfo(api_name,
                     "Inner dimensions don't match for einsum matmul");
      return false;
    }

    uint64_t k = k_a;

    // Determine data type.
    data_type_t dt;
    if (std::is_same<T, float>::value) {
      dt = data_type_t::f32;
    } else if (std::is_same<T, Eigen::bfloat16>::value) {
      dt = data_type_t::bf16;
    } else {
      LogZenDNNLInfo(api_name, "Unsupported data type for einsum");
      return false;
    }

    // Get pointers to tensor data.
    T* a_data = const_cast<T*>(a.flat<T>().data());
    T* b_data = const_cast<T*>(b.flat<T>().data());
    T* output_data = output->flat<T>().data();

    if (use_direct_api) {
      // Direct API path (lowoha::matmul::matmul_direct).

      // Calculate leading dimensions for row-major layout.
      int lda = transpose_a ? static_cast<int>(m) : static_cast<int>(k);
      int ldb = transpose_b ? static_cast<int>(k) : static_cast<int>(n);
      int ldc = static_cast<int>(n);

      // Setup data types.
      matmul_data_types dtypes;
      dtypes.src = dt;
      dtypes.wei = dt;
      dtypes.dst = dt;
      dtypes.bias = dt;
      dtypes.compute = dt;

      // Setup matmul_params.
      matmul_params params;
      params.dtypes = dtypes;
      params.mem_format_a = 'n';
      params.mem_format_b = 'n';
      params.lowoha_algo = matmul_algo_t::none;

      // Create empty batch params for 2D matmul
      matmul_batch_params_t batch_params;

      // Call matmul_direct.
      status_t status = matmul_direct(
          'r',                                    // Row-major layout
          transpose_a,                            // Transpose A flag
          transpose_b,                            // Transpose B flag
          m, n, k,                                // Dimensions
          1.0f,                                   // Alpha
          static_cast<const void*>(a_data), lda,  // Source A
          static_cast<const void*>(b_data), ldb,  // Source B (weights)
          nullptr,  // Bias (none - not implemented yet)
          0.0f,     // Beta
          static_cast<void*>(output_data), ldc,  // Destination
          true,                                  // is_weights_const
          batch_params,                          // No batch params for 2D
          params);

      if (status == status_t::success) {
        LogZenDNNLSuccess(api_name);
        return true;
      } else {
        LogZenDNNLInfo(api_name, ("Execution failed with status " +
                                  std::to_string(static_cast<int>(status)))
                                     .c_str());
        return false;
      }
    } else {
      // Operator API path.
      uint64_t a_buffer_size = a.NumElements() * sizeof(T);
      uint64_t b_buffer_size = b.NumElements() * sizeof(T);
      uint64_t out_buffer_size = output->NumElements() * sizeof(T);

      // Create tensors.
      tensor_t input_a, input_b, output_tensor;

      // Setup input A tensor.
      std::string a_order = transpose_a ? "ba" : "ab";
      input_a.set_size({m, k})
          .set_data_type(dt)
          .set_order(a_order)
          .set_storage(static_cast<void*>(a_data), a_buffer_size)
          .create();

      // Setup input B tensor.
      std::string b_order = transpose_b ? "ba" : "ab";
      input_b.set_size({k, n})
          .set_data_type(dt)
          .set_order(b_order)
          .set_storage(static_cast<void*>(b_data), b_buffer_size)
          .set_name("weights")
          .create();

      // Setup output tensor.
      output_tensor.set_size({m, n})
          .set_data_type(dt)
          .set_order("ab")
          .set_storage(static_cast<void*>(output_data), out_buffer_size)
          .create();

      // Create matmul context.
      matmul_context_t matmul_context;
      matmul_context.set_param("weights", input_b);
      matmul_context.create();

      // Create and execute matmul operator.
      matmul_operator_t matmul_operator;
      matmul_operator.set_name("tf_zendnnl_einsum_matmul")
          .set_context(matmul_context)
          .create();

      input_a.set_name("matmul_input");
      output_tensor.set_name("matmul_output");

      matmul_operator.set_input("matmul_input", input_a);
      matmul_operator.set_output("matmul_output", output_tensor);

      status_t status = matmul_operator.execute();

      if (status != status_t::success) {
        LogZenDNNLInfo(api_name, ("Execution failed with status " +
                                  std::to_string(static_cast<int>(status)))
                                     .c_str());
        return false;
      }

      LogZenDNNLSuccess(api_name);
      return true;
    }
  } catch (const zendnnl::error_handling::exception_t& e) {
    LogZenDNNLFallback("_ZenEinsum",
                       ("ZenDNNL exception: " + std::string(e.what())).c_str());
    return false;
  } catch (const std::exception& e) {
    LogZenDNNLFallback("_ZenEinsum",
                       ("Exception: " + std::string(e.what())).c_str());
    return false;
  }
}

// Explicit template instantiations.
template bool TryExecuteZenDNNLEinsumAsMatMul<float>(OpKernelContext*,
                                                     const Tensor&,
                                                     const Tensor&, Tensor*,
                                                     bool, bool);
template bool TryExecuteZenDNNLEinsumAsMatMul<Eigen::bfloat16>(
    OpKernelContext*, const Tensor&, const Tensor&, Tensor*, bool, bool);

// Execute einsum pattern abc,bcd->abd using ZenDNN's batch matmul.
// This pattern appears in MoE/expert projection layers.
// Input1: [a, b, c], Input2: [b, c, d] -> Output: [a, b, d]
// We iterate over dimension 'a' and perform b batch matmuls for each.
template <typename T>
bool TryExecuteZenDNNLEinsumAsBatchMatMulBroadcast(
    OpKernelContext* context,
    const Tensor& input1,  // [a, b, c]
    const Tensor& input2,  // [b, c, d]
    Tensor* output) {      // [a, b, d]
  try {
    using namespace zendnnl::common;
    using namespace zendnnl::memory;
    using namespace zendnnl::ops;
    using namespace zendnnl::lowoha::matmul;

    const bool use_direct_api = IsZenDnnMatmulDirectEnabled();
    const char* api_name = use_direct_api ? "_ZenEinsum(BatchBcast/Direct)"
                                          : "_ZenEinsum(BatchBcast)";

    // Get tensor shapes
    auto shape1 = input1.shape();
    auto shape2 = input2.shape();

    // Validate dimensions
    if (shape1.dims() != 3 || shape2.dims() != 3) {
      LogZenDNNLInfo(api_name, "abc,bcd->abd pattern requires 3D inputs");
      return false;
    }

    // Extract dimensions
    // Input1: [a, b, c]
    int64_t dim_a = shape1.dim_size(0);
    int64_t dim_b1 = shape1.dim_size(1);
    int64_t dim_c1 = shape1.dim_size(2);

    // Input2: [b, c, d]
    int64_t dim_b2 = shape2.dim_size(0);
    int64_t dim_c2 = shape2.dim_size(1);
    int64_t dim_d = shape2.dim_size(2);

    // Validate matching dimensions
    if (dim_b1 != dim_b2) {
      LogZenDNNLInfo(api_name, "Dimension 'b' mismatch between inputs");
      return false;
    }
    if (dim_c1 != dim_c2) {
      LogZenDNNLInfo(api_name,
                     "Dimension 'c' (contraction) mismatch between inputs");
      return false;
    }

    int64_t dim_b = dim_b1;
    int64_t dim_c = dim_c1;

    // For abc,bcd->abd:
    // We can reshape this as a batch matmul:
    // Reshape input1 [a, b, c] -> [a*b, 1, c] (batch of row vectors)
    // Reshape input2 [b, c, d] -> broadcast to [a*b, c, d] (with stride tricks)
    // Result [a*b, 1, d] -> reshape to [a, b, d]
    //
    // Alternative approach: iterate over 'a' and do batch matmul for each
    // This is simpler and avoids complex broadcasting in ZenDNN

    // Determine data type
    data_type_t dt;
    if (std::is_same<T, float>::value) {
      dt = data_type_t::f32;
    } else if (std::is_same<T, Eigen::bfloat16>::value) {
      dt = data_type_t::bf16;
    } else {
      LogZenDNNLInfo(api_name, "Unsupported data type");
      return false;
    }

    // Get data pointers
    const T* input1_data = input1.flat<T>().data();
    const T* input2_data = input2.flat<T>().data();
    T* output_data = output->flat<T>().data();

    // Strides for navigation
    int64_t input1_stride_a =
        dim_b * dim_c;  // stride to next 'a' slice in input1
    int64_t output_stride_a =
        dim_b * dim_d;  // stride to next 'a' slice in output

    if (use_direct_api) {
      // For each value of 'a', we do a batch matmul:
      // input1[a, :, :] shape [b, c] @ input2[:, :, :] shape [b, c, d]
      // But input1[a,:,:] is [b, c] and input2 is [b, c, d]
      // This is: for each b: input1[a, b, :] (1, c) @ input2[b, :, :] (c, d) ->
      // (1, d) So effectively: [b, 1, c] @ [b, c, d] -> [b, 1, d] squeezed to
      // [b, d]
      //
      // Actually simpler: treat input1[a,:,:] as [b, c] matrix
      // and for each b: row_vector[c] @ matrix[c, d] -> row_vector[d]
      // This is exactly batch matmul with batch size = b

      // Setup matmul params
      matmul_data_types dtypes;
      dtypes.src = dt;
      dtypes.wei = dt;
      dtypes.dst = dt;
      dtypes.bias = dt;
      dtypes.compute = dt;

      matmul_params params;
      params.dtypes = dtypes;
      params.mem_format_a = 'n';
      params.mem_format_b = 'n';
      params.lowoha_algo = matmul_algo_t::none;

      // For this pattern, we'll do a loop over 'a' and use batch matmul for 'b'
      // Each iteration: input1_slice[b, c] treated as batch of row vectors
      //                 input2[b, c, d] as batch of matrices
      // Result: output_slice[b, d]

      // Batch params for the b dimension
      matmul_batch_params_t batch_params;
      batch_params.Batch_A = static_cast<int>(dim_b);
      batch_params.Batch_B = static_cast<int>(dim_b);
      batch_params.batch_stride_src = dim_c;  // stride between batches in A
      batch_params.batch_stride_wei =
          dim_c * dim_d;  // stride between batches in B
      batch_params.batch_stride_dst =
          dim_d;  // stride between batches in output

      // M = a  (process all 'a' rows in ONE matmul per b), K=c, N=d
      uint64_t M = static_cast<uint64_t>(dim_a);
      uint64_t K = static_cast<uint64_t>(dim_c);
      uint64_t N = static_cast<uint64_t>(dim_d);

      // Leading dimensions — A and C now span ACROSS the 'b' axis
      const int lda = static_cast<int>(dim_b * dim_c);  // row stride of A = b·c
      const int ldb = static_cast<int>(dim_d);          // row stride of B = d
      const int ldc = static_cast<int>(dim_b * dim_d);  // row stride of C = b·d

      // ONE call replaces the entire for-a loop
      status_t status =
          matmul_direct('r', false, false, M, N, K, 1.0f,
                        static_cast<const void*>(input1_data), lda,
                        static_cast<const void*>(input2_data), ldb, nullptr,
                        0.0f, static_cast<void*>(output_data), ldc,
                        /*is_weights_const=*/true, batch_params, params);

      if (status != status_t::success) {
        LogZenDNNLInfo(api_name, "Batch matmul failed");
        return false;
      }

      LogZenDNNLSuccess(api_name);
      return true;

    } else {
      // Operator API path - similar logic with tensor objects
      // For simplicity, use the same loop approach

      uint64_t M = 1;
      uint64_t K = static_cast<uint64_t>(dim_c);
      uint64_t N = static_cast<uint64_t>(dim_d);

      for (int64_t a = 0; a < dim_a; ++a) {
        const T* src_ptr = input1_data + a * input1_stride_a;
        T* dst_ptr = output_data + a * output_stride_a;

        // Create tensors for this slice
        tensor_t input_a_tensor, input_b_tensor, output_tensor;

        // Input A: [b, c] -> batch of [1, c] row vectors
        uint64_t a_buffer_size = dim_b * dim_c * sizeof(T);
        input_a_tensor.set_size({static_cast<uint64_t>(dim_b), M, K})
            .set_data_type(dt)
            .set_order("abc")
            .set_storage(const_cast<void*>(static_cast<const void*>(src_ptr)),
                         a_buffer_size)
            .create();

        // Input B: [b, c, d]
        uint64_t b_buffer_size = dim_b * dim_c * dim_d * sizeof(T);
        input_b_tensor.set_size({static_cast<uint64_t>(dim_b), K, N})
            .set_data_type(dt)
            .set_order("abc")
            .set_storage(
                const_cast<void*>(static_cast<const void*>(input2_data)),
                b_buffer_size)
            .set_name("weights")
            .create();

        // Output: [b, d] -> batch of [1, d] row vectors
        uint64_t out_buffer_size = dim_b * dim_d * sizeof(T);
        output_tensor.set_size({static_cast<uint64_t>(dim_b), M, N})
            .set_data_type(dt)
            .set_order("abc")
            .set_storage(static_cast<void*>(dst_ptr), out_buffer_size)
            .create();

        // Create and execute matmul
        matmul_context_t matmul_context;
        matmul_context.set_param("weights", input_b_tensor);
        matmul_context.create();

        matmul_operator_t matmul_operator;
        matmul_operator.set_name("tf_zendnnl_einsum_bcast")
            .set_context(matmul_context)
            .create();

        input_a_tensor.set_name("matmul_input");
        output_tensor.set_name("matmul_output");

        matmul_operator.set_input("matmul_input", input_a_tensor);
        matmul_operator.set_output("matmul_output", output_tensor);

        status_t status = matmul_operator.execute();

        if (status != status_t::success) {
          LogZenDNNLInfo(
              api_name, ("Execution failed at a=" + std::to_string(a)).c_str());
          return false;
        }
      }

      LogZenDNNLSuccess(api_name);
      return true;
    }

  } catch (const zendnnl::error_handling::exception_t& e) {
    LogZenDNNLFallback("_ZenEinsum(BatchBcast)",
                       ("ZenDNNL exception: " + std::string(e.what())).c_str());
    return false;
  } catch (const std::exception& e) {
    LogZenDNNLFallback("_ZenEinsum(BatchBcast)",
                       ("Exception: " + std::string(e.what())).c_str());
    return false;
  }
}

// Explicit template instantiations for batch matmul broadcast.
template bool TryExecuteZenDNNLEinsumAsBatchMatMulBroadcast<float>(
    OpKernelContext*, const Tensor&, const Tensor&, Tensor*);
template bool TryExecuteZenDNNLEinsumAsBatchMatMulBroadcast<Eigen::bfloat16>(
    OpKernelContext*, const Tensor&, const Tensor&, Tensor*);

// ZenEinsumOp kernel class.
// Handles einsum operations by decomposing supported patterns into ZenDNN
// primitives.
template <typename T>
class ZenEinsumOp : public OpKernel {
 public:
  explicit ZenEinsumOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("equation", &equation_));

    // Analyze the einsum pattern at construction time.
    decomposition_ = AnalyzeEinsumPattern(equation_);

    // Log pattern recognition result.
    if (decomposition_.pattern == EinsumPatternType::kUnsupported) {
      zendnnl::error_handling::apilog_warning(
          "_ZenEinsum: Unsupported pattern '", equation_,
          "' - this should not have been routed to _ZenEinsum");
    } else {
      zendnnl::error_handling::apilog_info("_ZenEinsum: Recognized pattern '",
                                           equation_, "'");
    }
  }

  void Compute(OpKernelContext* context) override {
    zendnnl::error_handling::apilog_info(
        "Executing _ZenEinsum Compute, equation='", equation_, "'");

    // The layout pass should only route supported patterns here.
    // If we get an unsupported pattern, it's a bug in the layout pass.
    OP_REQUIRES(context,
                decomposition_.pattern != EinsumPatternType::kUnsupported,
                errors::InvalidArgument(
                    "_ZenEinsum received unsupported equation: ", equation_,
                    ". This is a bug - unsupported patterns should not be "
                    "routed to _ZenEinsum."));

    // Handle based on pattern type.
    switch (decomposition_.pattern) {
      case EinsumPatternType::kMatMul:
        ComputeMatMulPattern(context);
        break;

      case EinsumPatternType::kBatchMatMulBroadcast:
        ComputeBatchMatMulBroadcastPattern(context);
        break;

      default:
        OP_REQUIRES(context, false,
                    errors::Internal("Unhandled einsum pattern type"));
    }

    zendnnl::error_handling::apilog_info("_ZenEinsum Compute completed");
  }

 private:
  std::string equation_;
  EinsumDecomposition decomposition_;

  // Compute handler for ij,jk->ik (2D matmul) pattern.
  void ComputeMatMulPattern(OpKernelContext* context) {
    // Get input tensors.
    OP_REQUIRES(context, context->num_inputs() == 2,
                errors::InvalidArgument(
                    "MatMul einsum pattern requires exactly 2 inputs, got ",
                    context->num_inputs()));

    const Tensor& a = context->input(0);
    const Tensor& b = context->input(1);

    // Validate inputs are 2D matrices.
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(a.shape()),
                errors::InvalidArgument("First input is not a matrix. Shape: ",
                                        a.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(b.shape()),
                errors::InvalidArgument("Second input is not a matrix. Shape: ",
                                        b.shape().DebugString()));

    // For ij,jk->ik: a[i,j] @ b[j,k] -> output[i,k]
    // No transpose needed for standard matmul.
    int64_t m = a.dim_size(0);   // i dimension
    int64_t k1 = a.dim_size(1);  // j dimension from a
    int64_t k2 = b.dim_size(0);  // j dimension from b
    int64_t n = b.dim_size(1);   // k dimension

    // Validate inner dimensions match.
    OP_REQUIRES(
        context, k1 == k2,
        errors::InvalidArgument("Inner dimensions do not match: a.dim(1)=", k1,
                                " vs b.dim(0)=", k2));

    // Allocate output tensor.
    TensorShape out_shape({m, n});
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    // Handle empty tensors.
    if (output->NumElements() == 0) {
      return;
    }

    if (a.NumElements() == 0 || b.NumElements() == 0) {
      // Fill output with zeros for empty inputs.
      functor::SetZeroFunctor<CPUDevice, T> f;
      f(context->eigen_cpu_device(), output->flat<T>());
      return;
    }

    // Execute via ZenDNN.
    bool success = TryExecuteZenDNNLEinsumAsMatMul<T>(
        context, a, b, output, decomposition_.transpose_a,
        decomposition_.transpose_b);

    OP_REQUIRES(context, success,
                errors::Internal("_ZenEinsum matmul execution failed"));
  }

  // Compute handler for abc,bcd->abd (batch matmul with broadcast) pattern.
  void ComputeBatchMatMulBroadcastPattern(OpKernelContext* context) {
    // Get input tensors.
    OP_REQUIRES(context, context->num_inputs() == 2,
                errors::InvalidArgument("BatchMatMulBroadcast einsum pattern "
                                        "requires exactly 2 inputs, got ",
                                        context->num_inputs()));

    const Tensor& input1 = context->input(0);  // [a, b, c]
    const Tensor& input2 = context->input(1);  // [b, c, d]

    // Validate inputs are 3D tensors.
    OP_REQUIRES(
        context, input1.dims() == 3,
        errors::InvalidArgument("First input must be 3D [a,b,c]. Shape: ",
                                input1.shape().DebugString()));
    OP_REQUIRES(
        context, input2.dims() == 3,
        errors::InvalidArgument("Second input must be 3D [b,c,d]. Shape: ",
                                input2.shape().DebugString()));

    // Extract dimensions: abc,bcd->abd
    int64_t dim_a = input1.dim_size(0);
    int64_t dim_b1 = input1.dim_size(1);
    int64_t dim_c1 = input1.dim_size(2);
    int64_t dim_b2 = input2.dim_size(0);
    int64_t dim_c2 = input2.dim_size(1);
    int64_t dim_d = input2.dim_size(2);

    // Validate matching dimensions.
    OP_REQUIRES(context, dim_b1 == dim_b2,
                errors::InvalidArgument(
                    "Dimension 'b' mismatch: input1.dim(1)=", dim_b1,
                    " vs input2.dim(0)=", dim_b2));
    OP_REQUIRES(context, dim_c1 == dim_c2,
                errors::InvalidArgument(
                    "Dimension 'c' (contraction) mismatch: input1.dim(2)=",
                    dim_c1, " vs input2.dim(1)=", dim_c2));

    // Allocate output tensor [a, b, d].
    TensorShape out_shape({dim_a, dim_b1, dim_d});
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    // Handle empty tensors.
    if (output->NumElements() == 0) {
      return;
    }

    if (input1.NumElements() == 0 || input2.NumElements() == 0) {
      functor::SetZeroFunctor<CPUDevice, T> f;
      f(context->eigen_cpu_device(), output->flat<T>());
      return;
    }

    // Execute via ZenDNN.
    bool success = TryExecuteZenDNNLEinsumAsBatchMatMulBroadcast<T>(
        context, input1, input2, output);

    OP_REQUIRES(
        context, success,
        errors::Internal("_ZenEinsum batch matmul broadcast execution failed"));
  }
};

// Register kernels for supported data types.
#define REGISTER_EINSUM_KERNELS(TYPE)                                  \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("_ZenEinsum").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      ZenEinsumOp<TYPE>);

TF_CALL_float(REGISTER_EINSUM_KERNELS);
TF_CALL_bfloat16(REGISTER_EINSUM_KERNELS);
#undef REGISTER_EINSUM_KERNELS

}  // namespace amd_cpu_plugin
