/*******************************************************************************
 * Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 ******************************************************************************/

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <limits>
#include <memory>
#include <vector>

#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/fill_functor.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/fused_eigen_output_kernels.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_kernel_common.h"
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_zendnnl_utils.h"
#include "tensorflow_plugin/src/amd_cpu/util/errors.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_kernel.h"
#include "tensorflow_plugin/src/amd_cpu/util/op_requires.h"
#include "tensorflow_plugin/src/amd_cpu/util/padding.h"
#include "tensorflow_plugin/src/amd_cpu/util/register_types.h"
#include "tensorflow_plugin/src/amd_cpu/util/tensor_format.h"
#include "zendnnl.hpp"

namespace amd_cpu_plugin {

typedef Eigen::ThreadPoolDevice CPUDevice;

// Unified ZenDNNL MatMul implementation.
// Uses if-else to select between Direct API (lowoha::matmul_direct) and
// Operator API (matmul_operator_t) based on USE_ZENDNN_MATMUL_DIRECT env var.
template <typename T>
bool TryExecuteZenDNNLMatMul(OpKernelContext *context, const Tensor &a,
                             const Tensor &b, const Tensor *bias,
                             Tensor *output, bool transpose_a, bool transpose_b,
                             FusedComputationType fusion_type,
                             const Tensor *addend = nullptr) {
  try {
    using namespace zendnnl::memory;
    using namespace zendnnl::ops;
    using namespace zendnnl::lowoha::matmul;

    // Check which API path to use.
    const bool use_direct_api = IsZenDnnMatmulDirectEnabled();
    const char *api_name = use_direct_api ? "MatMulDirect" : "MatMul";

    // Get tensor dimensions.
    auto a_shape = a.shape();
    auto b_shape = b.shape();

    // Basic validation for 2D MatMul.
    if (a_shape.dims() != 2 || b_shape.dims() != 2) {
      LogZenDNNLInfo(api_name, "Only 2D MatMul supported");
      return false;
    }

    // Calculate dimensions - use proper unsigned types.
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
      LogZenDNNLInfo(api_name, "Inner dimensions don't match");
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
      LogZenDNNLInfo(api_name, "Unsupported data type");
      return false;
    }

    // Get pointers to TensorFlow tensor data.
    T *a_data = const_cast<T *>(a.flat<T>().data());
    T *b_data = const_cast<T *>(b.flat<T>().data());
    T *output_data = output->flat<T>().data();

    const bool should_use_bias =
        bias && fusion_type != FusedComputationType::kUndefined &&
        fusion_type != FusedComputationType::kRelu;

    // Check which API path to use.
    if (use_direct_api) {
      // Direct API path (lowoha::matmul::matmul_direct).

      // Get bias pointer if needed.
      const void *bias_ptr =
          should_use_bias ? static_cast<const void *>(bias->flat<T>().data())
                          : nullptr;

      // Calculate leading dimensions for row-major layout.
      int lda = transpose_a ? static_cast<int>(m) : static_cast<int>(k);
      int ldb = transpose_b ? static_cast<int>(k) : static_cast<int>(n);
      int ldc = static_cast<int>(n);

      // Setup data types for lowoha.
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

      // Setup post-ops based on fusion type.
      switch (fusion_type) {
        case FusedComputationType::kBiasAddWithRelu:
        case FusedComputationType::kRelu: {
          matmul_post_op relu_op;
          relu_op.po_type = post_op_type_t::relu;
          params.postop_.push_back(relu_op);
          break;
        }
        case FusedComputationType::kBiasAddWithSigmoid: {
          matmul_post_op sigmoid_op;
          sigmoid_op.po_type = post_op_type_t::sigmoid;
          params.postop_.push_back(sigmoid_op);
          break;
        }
        case FusedComputationType::kBiasAddWithTanh: {
          matmul_post_op tanh_op;
          tanh_op.po_type = post_op_type_t::tanh;
          params.postop_.push_back(tanh_op);
          break;
        }
        case FusedComputationType::kBiasAddWithGeluApproximate: {
          matmul_post_op gelu_op;
          gelu_op.po_type = post_op_type_t::gelu_tanh;
          params.postop_.push_back(gelu_op);
          break;
        }
        case FusedComputationType::kBiasAddWithGeluExact: {
          matmul_post_op gelu_op;
          gelu_op.po_type = post_op_type_t::gelu_erf;
          params.postop_.push_back(gelu_op);
          break;
        }
        case FusedComputationType::kBiasAddWithAdd: {
          if (addend) {
            matmul_post_op add_op;
            add_op.po_type = post_op_type_t::binary_add;
            add_op.buff = const_cast<void *>(
                static_cast<const void *>(addend->flat<T>().data()));
            add_op.dtype = dtypes.dst;
            add_op.dims = {static_cast<int64_t>(m), static_cast<int64_t>(n)};
            add_op.alpha = 1.0f;
            params.postop_.push_back(add_op);
          } else {
            LogZenDNNLInfo(
                api_name, "Binary add requested but no addend tensor provided");
            return false;
          }
          break;
        }
        case FusedComputationType::kBiasAddWithAddAndRelu: {
          if (addend) {
            matmul_post_op add_op;
            add_op.po_type = post_op_type_t::binary_add;
            add_op.buff = const_cast<void *>(
                static_cast<const void *>(addend->flat<T>().data()));
            add_op.dtype = dtypes.dst;
            add_op.dims = {static_cast<int64_t>(m), static_cast<int64_t>(n)};
            add_op.alpha = 1.0f;
            params.postop_.push_back(add_op);

            matmul_post_op relu_op;
            relu_op.po_type = post_op_type_t::relu;
            params.postop_.push_back(relu_op);
          } else {
            LogZenDNNLInfo(
                api_name,
                "Binary add+relu requested but no addend tensor provided");
            return false;
          }
          break;
        }
        default:
          break;
      }

      matmul_batch_params_t batch_params{};
      // TODO: Check for constant weight for performance optimization.
      // TODO: Currently handling 2D matrices only (Batch_A = 1, Batch_B = 1 as
      // per lowoha_common.hpp default constructor). For future 3D matrix
      // support, batch parameters must be properly configured.
      auto status = matmul_direct(
          'r' /* row-major */, transpose_a, transpose_b, static_cast<int>(m),
          static_cast<int>(n), static_cast<int>(k), 1.0f /* alpha */,
          static_cast<const void *>(a_data), lda,
          static_cast<const void *>(b_data), ldb, bias_ptr, 0.0f /* beta */,
          static_cast<void *>(output_data), ldc, false /* is_weights_const */,
          batch_params, params);

      if (status != zendnnl::error_handling::status_t::success) {
        LogZenDNNLInfo(api_name, ("Execution failed with status " +
                                  std::to_string(static_cast<int>(status)))
                                     .c_str());
        return false;
      }

    } else {
      // Operator API path (matmul_operator_t).
      using namespace zendnnl::common;

      // Calculate buffer sizes.
      uint64_t a_buffer_size = a.NumElements() * sizeof(T);
      uint64_t b_buffer_size = b.NumElements() * sizeof(T);
      uint64_t out_buffer_size = output->NumElements() * sizeof(T);

      // Create input A tensor.
      tensor_t input_a;
      input_a.set_size({m, k})
          .set_data_type(dt)
          .set_order(transpose_a ? "ba" : "ab")
          .set_storage(static_cast<void *>(a_data), a_buffer_size)
          .create();

      // Create output tensor.
      tensor_t output_tensor;
      output_tensor.set_size({m, n})
          .set_data_type(dt)
          .set_storage(static_cast<void *>(output_data), out_buffer_size)
          .create();

      // Create weights tensor (matrix B).
      tensor_t weights;
      weights.set_size({k, n})
          .set_data_type(dt)
          .set_order(transpose_b ? "ba" : "ab")
          .set_storage(static_cast<void *>(b_data), b_buffer_size)
          .set_name("weights")
          .create();

      // Create matmul context.
      matmul_context_t matmul_context;
      matmul_context.set_param("weights", weights);

      // Handle bias if present.
      tensor_t bias_tensor;
      if (should_use_bias) {
        T *bias_data = const_cast<T *>(bias->flat<T>().data());
        uint64_t bias_buffer_size = bias->NumElements() * sizeof(T);

        // ZenDNNL requires bias dimensions to match output dimensions
        // For 2D matmul output {m, n}, bias must be {1, n}.
        std::vector<uint64_t> bias_dims = {1, n};
        bias_tensor.set_size(bias_dims)
            .set_data_type(dt)
            .set_storage(static_cast<void *>(bias_data), bias_buffer_size)
            .set_name("bias")
            .create();

        matmul_context.set_param("bias", bias_tensor);
      }

      // Add post-ops based on fusion type.
      tensor_t addend_tensor;
      switch (fusion_type) {
        case FusedComputationType::kBiasAddWithRelu:
        case FusedComputationType::kRelu: {
          post_op_t relu_post_op(post_op_type_t::relu);
          matmul_context.set_post_op(relu_post_op);
          break;
        }
        case FusedComputationType::kBiasAddWithTanh: {
          post_op_t tanh_post_op(post_op_type_t::tanh);
          matmul_context.set_post_op(tanh_post_op);
          break;
        }
        case FusedComputationType::kBiasAddWithSigmoid: {
          post_op_t sigmoid_post_op(post_op_type_t::sigmoid);
          matmul_context.set_post_op(sigmoid_post_op);
          break;
        }
        case FusedComputationType::kBiasAddWithGeluApproximate: {
          post_op_t gelu_post_op(post_op_type_t::gelu_tanh);
          matmul_context.set_post_op(gelu_post_op);
          break;
        }
        case FusedComputationType::kBiasAddWithGeluExact: {
          post_op_t gelu_post_op(post_op_type_t::gelu_erf);
          matmul_context.set_post_op(gelu_post_op);
          break;
        }
        case FusedComputationType::kBiasAddWithAdd: {
          if (addend) {
            T *addend_data = const_cast<T *>(addend->flat<T>().data());
            uint64_t addend_buffer_size = addend->NumElements() * sizeof(T);

            addend_tensor.set_size({m, n})
                .set_data_type(dt)
                .set_storage(static_cast<void *>(addend_data),
                             addend_buffer_size)
                .set_name("binary_add_tensor_0")
                .create();

            binary_add_params_t add_params{1.0f, "binary_add_tensor_0"};
            post_op_t add_post_op(add_params);
            matmul_context.set_post_op(add_post_op);
            matmul_context.set_param("addend", addend_tensor);
          } else {
            LogZenDNNLInfo(
                api_name, "Binary add requested but no addend tensor provided");
            return false;
          }
          break;
        }
        case FusedComputationType::kBiasAddWithAddAndRelu: {
          if (addend) {
            T *addend_data = const_cast<T *>(addend->flat<T>().data());
            uint64_t addend_buffer_size = addend->NumElements() * sizeof(T);

            addend_tensor.set_size({m, n})
                .set_data_type(dt)
                .set_storage(static_cast<void *>(addend_data),
                             addend_buffer_size)
                .set_name("binary_add_tensor_0")
                .create();

            binary_add_params_t add_params{1.0f, "binary_add_tensor_0"};
            post_op_t add_post_op(add_params);
            matmul_context.set_post_op(add_post_op);

            post_op_t relu_post_op(post_op_type_t::relu);
            matmul_context.set_post_op(relu_post_op);

            matmul_context.set_param("addend", addend_tensor);
          } else {
            LogZenDNNLInfo(
                api_name,
                "Binary add+relu requested but no addend tensor provided");
            return false;
          }
          break;
        }
        default:
          break;
      }

      matmul_context.create();

      // Create and execute matmul operator.
      matmul_operator_t matmul_operator;
      matmul_operator.set_name("tf_zendnnl_matmul")
          .set_context(matmul_context)
          .create();

      input_a.set_name("matmul_input");
      output_tensor.set_name("matmul_output");

      // Execute MatMul operation.
      matmul_operator.set_input("matmul_input", input_a);
      matmul_operator.set_output("matmul_output", output_tensor);

      // Set addend tensor input for binary add post-ops.
      int post_op_count = matmul_context.get_post_op_count();
      for (int i = 0; i < post_op_count; i++) {
        post_op_t post_op = matmul_context.get_post_op(i);
        if (post_op.type == post_op_type_t::binary_add) {
          matmul_operator.set_input(post_op.binary_add_params.tensor_name,
                                    addend_tensor);
        }
      }

      status_t status = matmul_operator.execute();

      if (status != status_t::success) {
        LogZenDNNLInfo(api_name, ("Execution failed with status " +
                                  std::to_string(static_cast<int>(status)))
                                     .c_str());
        return false;
      }
    }

    return true;

  } catch (const zendnnl::error_handling::exception_t &e) {
    LogZenDNNLFallback("MatMul",
                       ("ZenDNNL exception: " + std::string(e.what())).c_str());
    return false;
  } catch (const std::exception &e) {
    LogZenDNNLFallback("MatMul",
                       ("Exception: " + std::string(e.what())).c_str());
    return false;
  }
}

// Explicit template instantiations.
template bool TryExecuteZenDNNLMatMul<float>(OpKernelContext *, const Tensor &,
                                             const Tensor &, const Tensor *,
                                             Tensor *, bool, bool,
                                             FusedComputationType,
                                             const Tensor *);
template bool TryExecuteZenDNNLMatMul<Eigen::bfloat16>(
    OpKernelContext *, const Tensor &, const Tensor &, const Tensor *, Tensor *,
    bool, bool, FusedComputationType, const Tensor *);

template <typename Device, typename T, bool is_bias_add_gelu = false,
          bool is_fused = false>
class ZenMatMulOp : public OpKernel {
 public:
  explicit ZenMatMulOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(context, context->GetAttr("transpose_b", &transpose_b_));

    std::vector<FusedComputationPattern> patterns;
    is_reshape_ = false;
    if (is_fused) {
      OP_REQUIRES_OK(context, context->GetAttr("is_reshape", &is_reshape_));
      using FCT = FusedComputationType;
      patterns = {
          {FCT::kBiasAdd, {"BiasAdd"}},
          {FCT::kBiasAddWithAdd, {"BiasAdd", "Add"}},
          {FCT::kBiasAddWithRelu, {"BiasAdd", "Relu"}},
          {FCT::kBiasAddWithSigmoid, {"BiasAdd", "Sigmoid"}},
          {FCT::kBiasAddWithTanh, {"BiasAdd", "Tanh"}},
          {FCT::kBiasAddWithGeluExact, {"BiasAdd", "GeluExact"}},
          {FCT::kBiasAddWithAddAndRelu, {"BiasAdd", "Add", "Relu"}},
          {FCT::kBiasAddWithGeluApproximate, {"BiasAdd", "GeluApproximate"}},
          {FCT::kRelu, {"Relu"}}};
      OP_REQUIRES_OK(context,
                     InitializeFusedComputation(context, "_ZenMatMul", patterns,
                                                &fused_computation_,
                                                &fused_computation_args_));
    }
  }

  void Compute(OpKernelContext *context) override {
    // Old ZenDNN logging removed;

    const Tensor &a = context->input(0);
    const Tensor &b = context->input(1);
    // Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(
        context, TensorShapeUtils::IsMatrix(a.shape()),
        errors::InvalidArgument("In[0] is not a matrix. Instead it has shape ",
                                a.shape().DebugString()));
    OP_REQUIRES(
        context, TensorShapeUtils::IsMatrix(b.shape()),
        errors::InvalidArgument("In[1] is not a matrix. Instead it has shape ",
                                b.shape().DebugString()));
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
    dim_pair[0].first = transpose_a_ ? 0 : 1;
    dim_pair[0].second = transpose_b_ ? 1 : 0;

    OP_REQUIRES(context,
                a.dim_size(dim_pair[0].first) == b.dim_size(dim_pair[0].second),
                errors::InvalidArgument("Matrix size-incompatible: In[0]: ",
                                        a.shape().DebugString(),
                                        ", In[1]: ", b.shape().DebugString()));
    int a_dim_remaining = 1 - dim_pair[0].first;
    int b_dim_remaining = 1 - dim_pair[0].second;
    TensorShape out_shape;
    if (is_reshape_) {
      out_shape = {1, a.dim_size(a_dim_remaining), b.dim_size(b_dim_remaining)};
    } else {
      out_shape = {a.dim_size(a_dim_remaining), b.dim_size(b_dim_remaining)};
    }

    Tensor *out = nullptr;

    if ((fused_computation_ == FusedComputationType::kBiasAddWithAdd) ||
        (fused_computation_ == FusedComputationType::kBiasAddWithAddAndRelu)) {
      const Tensor &add_tensor = context->input(3);
      context->set_output(0, add_tensor);
      out = context->mutable_output(0);
    } else {
      OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &out));

      if (out->NumElements() == 0) {
        // If a has shape [0, x] or b has shape [x, 0], the output shape
        // is a 0-element matrix, so there is nothing to do.
        return;
      }
    }

    if (a.NumElements() == 0 && b.NumElements() == 0) {
      // If a has shape [x, 0] and b has shape [0, y], the output shape is
      // [x, y] where x and y are non-zero, so we fill the output with zeros.
      // Set output to zero for empty inputs
      // Note: SetZeroFunctor may not be directly available in plugin
      // For now, just return as output is already allocated
      if (is_fused) {
        functor::SetZeroFunctor<Device, T> f;
        f(context->eigen_cpu_device(), out->flat<T>());
      }
      return;
    }

    // Try ZenDNNL implementation
    bool zendnnl_success = false;

    const Tensor *bias = nullptr;
    const Tensor *addend = nullptr;

    // Extract bias tensor for fusion types that need it
    if (is_fused && fused_computation_ != FusedComputationType::kRelu) {
      bias = &context->input(2);
    }

    // Extract addend tensor for binary add fusions.
    if (fused_computation_ == FusedComputationType::kBiasAddWithAdd ||
        fused_computation_ == FusedComputationType::kBiasAddWithAddAndRelu) {
      addend = &context->input(3);  // Addend is the 4th input (index 3).
    }

    zendnnl_success =
        TryExecuteZenDNNLMatMul<T>(context, a, b, bias, out, transpose_a_,
                                   transpose_b_, fused_computation_, addend);
    if (zendnnl_success) {
      LogZenDNNLSuccess("MatMul");
    } else {
      LogZenDNNLFallback("MatMul", "failed");
    }

    // If ZenDNNL execution failed, report error.
    if (!zendnnl_success) {
      OP_REQUIRES_OK(
          context, errors::Internal("ZenDNNL MatMul execution failed. Old "
                                    "ZenDNN fallback path has been removed."));
    }

    // Old ZenDNN logging removed;
  }

 private:
  bool is_reshape_ = false;
  bool transpose_a_ = false;
  bool transpose_b_ = false;

  FusedComputationType fused_computation_ = FusedComputationType::kUndefined;
  FusedComputationArgs fused_computation_args_;
};
#define REGISTER_MATMUL_KERNELS(T)                                             \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_ZenFusedMatMul").Device(DEVICE_CPU).TypeConstraint<T>("T"),       \
      ZenMatMulOp<CPUDevice, T, false, true>);                                 \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_ZenMatMul").Device(DEVICE_CPU).TypeConstraint<T>("T"),            \
      ZenMatMulOp<CPUDevice, T>);                                              \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("_ZenMatMulBiasAddGelu").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ZenMatMulOp<CPUDevice, T, true>);                                        \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("MatMulBiasAddGelu").Device(DEVICE_CPU).TypeConstraint<T>("T"),     \
      ZenMatMulOp<CPUDevice, T, true>);

TF_CALL_float(REGISTER_MATMUL_KERNELS);
TF_CALL_bfloat16(REGISTER_MATMUL_KERNELS);
#undef REGISTER_MATMUL_KERNELS

}  // namespace amd_cpu_plugin
