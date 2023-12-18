/*******************************************************************************
 * Modifications Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 ******************************************************************************/

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// TensorFlow plug-in headers.
#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_fused_batchnorm_kernel.h"

#include "tensorflow_plugin/src/amd_cpu/kernels/zendnn/zen_mempool.h"
#include "tensorflow_plugin/src/amd_cpu/util/register_types.h"

namespace amd_cpu_plugin {

// Adding a third parameter to the template to support FusedBatchNormV3.
// This is different from default where the classes are derived. Moves enabling
// to compile-time rather than runtime.
template <typename Device, typename T, typename U, bool reserved_space,
          bool is_batch_norm_ex = false>
class ZenFusedBatchNormOp : public OpKernel {
 public:
  explicit ZenFusedBatchNormOp(OpKernelConstruction *context)
      : OpKernel(context) {
    float epsilon;
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon));
    epsilon_ = U(epsilon);

    float exponential_avg_factor;
    OP_REQUIRES_OK(context, context->GetAttr("exponential_avg_factor",
                                             &exponential_avg_factor));
    exponential_avg_factor_ = U(exponential_avg_factor);

    string tensor_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &tensor_format));
    OP_REQUIRES(context, FormatFromString(tensor_format, &tensor_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("is_training", &is_training_));

    if (!is_batch_norm_ex) {
      activation_mode_ = FusedBNActivationMode::kIdentity;
    } else {
      int num_side_inputs;
      OP_REQUIRES_OK(context,
                     context->GetAttr("num_side_inputs", &num_side_inputs));
      OP_REQUIRES(context, num_side_inputs == 0,
                  errors::InvalidArgument(
                      "_ZenFusedBatchNorm do not support side input now."));

      OP_REQUIRES_OK(context, ParseActivationMode(context, &activation_mode_));
      OP_REQUIRES(context, activation_mode_ == FusedBNActivationMode::kRelu,
                  errors::InvalidArgument(
                      "_ZenFusedBatchNorm only support Relu activation"));
    }

    OP_REQUIRES_OK(context, InitZendnnParameters(context, &zendnn_params_));

    depth_ = 0;
    mean_values_ = nullptr;
    variance_values_ = nullptr;
  }

  void Compute(OpKernelContext *context) override {
    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-DEF: _ZenFusedBatchNorm (TF kernel): In Compute!");

    const size_t kSrcIndex = 0;       // Index of src input tensor.
    const size_t kScaleIndex = 1;     // Index of scale tensor.
    const size_t kShiftIndex = 2;     // Index of shift tensor.
    const size_t kMeanIndex = 3;      // Index of est_mean tensor.
    const size_t kVarianceIndex = 4;  // Index of est_variance tensor.

    const Tensor &src_tensor = context->input(kSrcIndex);
    const Tensor &scale_tensor = context->input(kScaleIndex);
    const Tensor &shift_tensor = context->input(kShiftIndex);
    const Tensor &est_mean_tensor = context->input(kMeanIndex);
    const Tensor &est_variance_tensor = context->input(kVarianceIndex);

    TensorShape tf_shape_src;
    tf_shape_src = src_tensor.shape();
    OP_REQUIRES(context, src_tensor.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        src_tensor.shape().DebugString()));
    OP_REQUIRES(context, scale_tensor.dims() == 1,
                errors::InvalidArgument("scale must be 1-dimensional",
                                        scale_tensor.shape().DebugString()));
    OP_REQUIRES(context, shift_tensor.dims() == 1,
                errors::InvalidArgument("offset must be 1-dimensional",
                                        shift_tensor.shape().DebugString()));
    OP_REQUIRES(context, est_mean_tensor.dims() == 1,
                errors::InvalidArgument("estimated_mean must be 1-dimensional",
                                        est_mean_tensor.shape().DebugString()));
    OP_REQUIRES(
        context, est_variance_tensor.dims() == 1,
        errors::InvalidArgument("estimated_variance must be 1-dimensional",
                                est_variance_tensor.shape().DebugString()));

    // Handle the special case: input with 0 element and 0 batch size.
    Tensor *dst_tensor = nullptr;
    TensorShape workspace_tf_shape;
    if (tf_shape_src.num_elements() == 0) {
      size_t workspace_bytes = 0;
      workspace_tf_shape.AddDim(workspace_bytes);
      HandleEmptyInput(context, tf_shape_src, workspace_tf_shape,
                       scale_tensor.shape(), &dst_tensor);
      return;
    }

    depth_ = static_cast<int>(GetTensorDim(src_tensor, tensor_format_, 'C'));

    // Index of output tensor.
    const size_t kDstIndex = 0;

    // Allocate 5 output TF tensors.
    Tensor *batch_mean_tensor = nullptr;
    Tensor *batch_variance_tensor = nullptr;
    Tensor *saved_mean_tensor = nullptr;
    Tensor *saved_variance_tensor = nullptr;
    Tensor *reserved_space_tensor = nullptr;

    memory::format_tag dnn_fmt;
    if (tensor_format_ == FORMAT_NHWC) {
      dnn_fmt = memory::format_tag::nhwc;
    } else if (tensor_format_ == FORMAT_NCHW) {
      dnn_fmt = memory::format_tag::nchw;
    } else {
      OP_REQUIRES(context, false,
                  errors::InvalidArgument("Unsupported data format"));
    }

    int inp_batch =
        tf_shape_src.dim_size(GetTensorDimIndex(tensor_format_, 'N'));
    int inp_depth =
        tf_shape_src.dim_size(GetTensorDimIndex(tensor_format_, 'C'));
    int inp_rows =
        tf_shape_src.dim_size(GetTensorDimIndex(tensor_format_, 'H'));
    int inp_cols =
        tf_shape_src.dim_size(GetTensorDimIndex(tensor_format_, 'W'));

    zendnnEnv zen_env_obj = readEnv();
    bool blocked = zen_env_obj.zenConvAlgo == zenConvAlgoType::DIRECT1 &&
                   !zendnn_params_.is_eager;
    bool blocked_nhwc = zen_env_obj.zenConvAlgo == zenConvAlgoType::DIRECT2;

    if (inp_depth % 8 != 0 && blocked && !blocked_nhwc) {
      OP_REQUIRES_OK(
          context,
          errors::Internal("ZENDNN_BLOCKED_FORMAT not supported for this "
                           "model, Please use another data format."));
    }

    // Set src memory descriptor.
    memory::dims src_dims =
        memory::dims({inp_batch, inp_depth, inp_rows, inp_cols});

    auto src_md = memory::desc(src_dims, memory::data_type::f32, dnn_fmt);

    ZenBatchNormFwdParams fwd_params(src_dims, depth_, epsilon_, is_training_,
                                     src_md, activation_mode_);

    // Get forward batch-normalization op from the primitive caching pool.
    ZenFusedBatchNormFwdPrimitive<T, U> *bn_fwd =
        ZenFusedBatchNormFwdPrimitiveFactory<T, U>::Get(fwd_params);

    // Allocate workspace tensor.
    U *ws_data = nullptr;
    if (fwd_params.activation_mode == FusedBNActivationMode::kRelu) {
      memory::desc workspace_md = bn_fwd->GetBatchNormFwdPd()->workspace_desc();
      size_t workspace_bytes = workspace_md.get_size();
      workspace_tf_shape.AddDim(workspace_bytes);

      AllocateTFOutputs(context, scale_tensor.shape(), workspace_tf_shape,
                        &batch_mean_tensor, &batch_variance_tensor,
                        &saved_mean_tensor, &saved_variance_tensor,
                        &reserved_space_tensor);
      if (reserved_space) {
        ws_data = static_cast<U *>(reserved_space_tensor->flat<U>().data());
      }
    } else {
      // There is actually no workspace tensor out, so we make a dummy one.
      size_t workspace_bytes = 0;
      workspace_tf_shape.AddDim(workspace_bytes);
      AllocateTFOutputs(context, scale_tensor.shape(), workspace_tf_shape,
                        &batch_mean_tensor, &batch_variance_tensor,
                        &saved_mean_tensor, &saved_variance_tensor,
                        &reserved_space_tensor);
    }

    if (is_training_) {
      SetMeanVariance(*batch_mean_tensor, *batch_variance_tensor);
    } else {
      SetMeanVariance(est_mean_tensor, est_variance_tensor);
    }

    // Pack scale & shift as "weights":
    // <scale>...<scale><shift>...<shift>
    Tensor weights_tensor;
    TensorShape weights_shape({2, depth_});
    OP_REQUIRES_OK(
        context, context->allocate_temp(DataTypeToEnum<U>::value, weights_shape,
                                        &weights_tensor));
    U *weights_data = weights_tensor.flat<U>().data();
    const U *scale_tf = scale_tensor.flat<U>().data();
    const U *shift_tf = shift_tensor.flat<U>().data();

    std::memcpy(weights_data, scale_tf, depth_ * sizeof(U));
    std::memcpy(weights_data + depth_, shift_tf, depth_ * sizeof(U));
    char *saved_mean_data_tf =
        reinterpret_cast<char *>(saved_mean_tensor->flat<U>().data());
    std::memcpy(saved_mean_data_tf, reinterpret_cast<char *>(mean_values_),
                depth_ * sizeof(U));

    char *saved_variance_data_tf =
        reinterpret_cast<char *>(saved_variance_tensor->flat<U>().data());
    std::memcpy(saved_variance_data_tf,
                reinterpret_cast<char *>(variance_values_), depth_ * sizeof(U));

    const T *src_data =
        static_cast<T *>(const_cast<T *>(src_tensor.flat<T>().data()));

    // Update the output type.
    ZenTensorType out_type = ZenTensorType::kFloat;

    // Allocate output (dst) tensor.
    TensorShape tf_shape_dst = tf_shape_src;

    // TODO(zendnn): Remove the hard reset on enabling ZenMemPool.
    // Enbaling ZenMemPool with zenEnableMemPool=2 affects the accuracy, but
    // provides 5-6% boost in performance for some sizes.
    // Hence after fixing the bug with zenEnableMemPool=2 enabling, the followng
    // hard resetting code shall be removed.
    int zen_enable_mempool =
        ((zen_env_obj.zenEnableMemPool == 2) || zendnn_params_.is_eager)
            ? 0
            : zen_env_obj.zenEnableMemPool;
    ZenMemoryPool<T> *zen_pool_buffer;

    // ZenMemPool Optimization reuse o/p tensors from the pool. By default its
    // enabled, export ZENDNN_ENABLE_MEMPOOL=0 will disable memory pool
    // optimization.
    // Cases where tensors in pool are not free or requested size is more than
    // available tensor size in Pool, control will fall back to default way of
    // allocation i.e. with allocate_output(..).
    if (zen_enable_mempool % MEMPOOL_TYPE) {
      unsigned int thread_id = GetZenTFthreadId(std::this_thread::get_id());
      zen_pool_buffer = ZenMemoryPool<T>::GetZenMemPool(thread_id);
      if (zen_pool_buffer) {
        int status = zen_pool_buffer->AcquireZenPoolTensor(
            context, &dst_tensor, tf_shape_dst, zendnn_params_.out_links,
            zendnn_params_.reset, out_type);
        if (status) {
          zen_enable_mempool = 0;
        }
      } else {
        zen_enable_mempool = 0;
      }
    }
    if (!(zen_enable_mempool % MEMPOOL_TYPE)) {
      OP_REQUIRES_OK(context, context->allocate_output(kDstIndex, tf_shape_dst,
                                                       &dst_tensor));
    }

    U *weights_op_data = weights_data;
    U *mean_op_data = saved_mean_tensor->flat<U>().data();
    U *variance_op_data = saved_variance_tensor->flat<U>().data();
    T *dst_data = dst_tensor->flat<T>().data();

    bn_fwd->Execute(src_data, weights_op_data, dst_data, mean_op_data,
                    variance_op_data, ws_data);
    float adjust_factor = 1.0;
    if (is_training_) {
      size_t orig_size = src_dims[0] * src_dims[2] * src_dims[3];
      size_t adjust_size = (orig_size > 1) ? (orig_size - 1) : 1;
      adjust_factor = (static_cast<float>(orig_size)) / adjust_size;
    }

    auto mean_data = reinterpret_cast<U *>(saved_mean_data_tf);
    auto variance_data = reinterpret_cast<U *>(saved_variance_data_tf);
    auto batch_mean_data = batch_mean_tensor->flat<U>().data();
    auto batch_variance_data = batch_variance_tensor->flat<U>().data();
    auto est_mean_data = est_mean_tensor.flat<U>().data();
    auto est_variance_data = est_variance_tensor.flat<U>().data();

    if (is_training_) {
      if (exponential_avg_factor_ == U(1.0)) {
        for (int k = 0; k < depth_; k++) {
          batch_mean_data[k] = mean_data[k];
          batch_variance_data[k] =
              static_cast<U>(adjust_factor) * variance_data[k];
        }
      } else {
        U one_minus_factor = U(1.0) - exponential_avg_factor_;
        for (int k = 0; k < depth_; k++) {
          batch_mean_data[k] = one_minus_factor * est_mean_data[k] +
                               exponential_avg_factor_ * mean_data[k];
          batch_variance_data[k] = one_minus_factor * est_variance_data[k] +
                                   exponential_avg_factor_ *
                                       static_cast<U>(adjust_factor) *
                                       variance_data[k];
        }
      }
    } else {
      std::memcpy(batch_mean_data, mean_data, depth_ * sizeof(U));
      std::memcpy(batch_variance_data, variance_data, depth_ * sizeof(U));
    }
    if ((zen_env_obj.zenEnableMemPool % MEMPOOL_TYPE) &&
        !zendnn_params_.is_eager) {
      unsigned int thread_id = GetZenTFthreadId(std::this_thread::get_id());
      zen_pool_buffer = ZenMemoryPool<T>::GetZenMemPool(thread_id);
      if (zen_pool_buffer) {
        auto src_tensor_map = src_tensor.tensor<float, 4>();
        const float *src_tensor_array = src_tensor_map.data();
        zen_pool_buffer->ZenMemPoolFree(context,
                                        const_cast<float *>(src_tensor_array));
      }
    }
    zendnnInfo(
        ZENDNN_FWKLOG,
        "ZEN-OP-DEF: _ZenFusedBatchNorm (TF kernel): Compute Is Successful!");
  }

 private:
  float epsilon_;
  U exponential_avg_factor_;
  TensorFormat tensor_format_;
  bool is_training_;
  U *mean_values_;
  U *variance_values_;
  size_t depth_;  // Batch normalization is performed for per channel.
  FusedBNActivationMode activation_mode_;
  ZendnnParameters zendnn_params_;

  void SetMeanVariance(const Tensor &mean, const Tensor &variance) {
    mean_values_ =
        reinterpret_cast<U *>(const_cast<U *>(mean.flat<U>().data()));
    variance_values_ =
        reinterpret_cast<U *>(const_cast<U *>(variance.flat<U>().data()));
  }

  void HandleEmptyInput(OpKernelContext *context, TensorShape tf_shape_src,
                        TensorShape workspace_tf_shape,
                        TensorShape tf_shape_scale, Tensor **dst_tensor) {
    DCHECK(dst_tensor);
    const size_t kDstIndex = 0;
    OP_REQUIRES_OK(
        context, context->allocate_output(kDstIndex, tf_shape_src, dst_tensor));
    DCHECK(*dst_tensor);
    memset(const_cast<char *>((*dst_tensor)->tensor_data().data()), 0,
           (*dst_tensor)->tensor_data().size());

    Tensor *batch_mean_tensor = nullptr;
    Tensor *batch_variance_tensor = nullptr;
    Tensor *saved_mean_tensor = nullptr;
    Tensor *saved_variance_tensor = nullptr;
    Tensor *reserved_space_tensor = nullptr;
    AllocateTFOutputs(context, tf_shape_scale, workspace_tf_shape,
                      &batch_mean_tensor, &batch_variance_tensor,
                      &saved_mean_tensor, &saved_variance_tensor,
                      &reserved_space_tensor);
  }

  void AllocateTFOutputs(OpKernelContext *context, TensorShape tf_shape_scale,
                         TensorShape workspace_tf_shape,
                         Tensor **batch_mean_tensor,
                         Tensor **batch_variance_tensor,
                         Tensor **saved_mean_tensor,
                         Tensor **saved_variance_tensor,
                         Tensor **reserved_space_tensor) {
    DCHECK(batch_mean_tensor);
    DCHECK(batch_variance_tensor);
    DCHECK(saved_mean_tensor);
    DCHECK(saved_variance_tensor);

    const size_t kBatchMeanIndex = 1;
    const size_t kBatchVarianceIndex = 2;
    const size_t kSavedMeanIndex = 3;
    const size_t kSavedVarianceIndex = 4;
    const size_t kReservedSpaceIndex = 5;

    int num_elements = tf_shape_scale.num_elements();

    // Allocate batch mean output tensor.
    OP_REQUIRES_OK(context,
                   context->allocate_output(kBatchMeanIndex, tf_shape_scale,
                                            batch_mean_tensor));
    DCHECK(*batch_mean_tensor);
    // Set NAN mean value in case of empty input tensor.
    auto batch_mean_data = (*batch_mean_tensor)->flat<U>().data();
    std::fill_n(batch_mean_data, num_elements, static_cast<U>(NAN));

    // Allocate batch variance output tensor.
    OP_REQUIRES_OK(context,
                   context->allocate_output(kBatchVarianceIndex, tf_shape_scale,
                                            batch_variance_tensor));
    DCHECK(*batch_variance_tensor);
    // Set NAN variance value in case of empty input tensor.
    auto batch_variance_data = (*batch_variance_tensor)->flat<U>().data();
    std::fill_n(batch_variance_data, num_elements, static_cast<U>(NAN));

    // Mean and variance (without Bessel's correction) saved for backward
    // computation to serve as pre-computed mean and variance.
    OP_REQUIRES_OK(context,
                   context->allocate_output(kSavedMeanIndex, tf_shape_scale,
                                            saved_mean_tensor));
    DCHECK(*saved_mean_tensor);
    // Set 0 mean value in case of empty input tensor.
    auto saved_mean_data = (*saved_mean_tensor)->flat<U>().data();
    std::fill_n(saved_mean_data, num_elements, static_cast<U>(0));

    OP_REQUIRES_OK(context,
                   context->allocate_output(kSavedVarianceIndex, tf_shape_scale,
                                            saved_variance_tensor));
    DCHECK(*saved_variance_tensor);
    // Set 0 variance value in case of empty input tensor.
    auto saved_variance_data = (*saved_variance_tensor)->flat<U>().data();
    std::fill_n(saved_variance_data, num_elements, static_cast<U>(0));

    // Changes to support reserved_space_3 parameter in FusedBatchNormV3.
    if (reserved_space) {
      DCHECK(reserved_space_tensor != nullptr);
      OP_REQUIRES_OK(context, context->allocate_output(kReservedSpaceIndex,
                                                       workspace_tf_shape,
                                                       reserved_space_tensor));
      DCHECK((*reserved_space_tensor) != nullptr);
    }
  }
};

#define REGISTER_ZEN_FUSED_BATCHNORM_CPU(T)                                 \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("_ZenFusedBatchNorm").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ZenFusedBatchNormOp<CPUDevice, T, T, false, false>);

TF_CALL_float(REGISTER_ZEN_FUSED_BATCHNORM_CPU);
#undef REGISTER_ZEN_FUSED_BATCHNORM_CPU

#define REGISTER_ZEN_FUSED_BATCHNORM_V2_CPU(T, U)      \
  REGISTER_KERNEL_BUILDER(Name("_ZenFusedBatchNormV2") \
                              .Device(DEVICE_CPU)      \
                              .TypeConstraint<T>("T")  \
                              .TypeConstraint<U>("U"), \
                          ZenFusedBatchNormOp<CPUDevice, T, U, false, false>);

REGISTER_ZEN_FUSED_BATCHNORM_V2_CPU(float, float);
#undef REGISTER_ZEN_FUSED_BATCHNORM_V2_CPU

// TODO(zendnn) : FusedBatchNormV3 has an additional output that is used to hold
// intermediate results. This parameter functionality is not implemented on CPU.
#define REGISTER_ZEN_FUSED_BATCHNORM_V3_CPU(T, U)      \
  REGISTER_KERNEL_BUILDER(Name("_ZenFusedBatchNormV3") \
                              .Device(DEVICE_CPU)      \
                              .TypeConstraint<T>("T")  \
                              .TypeConstraint<U>("U"), \
                          ZenFusedBatchNormOp<CPUDevice, T, U, true, false>);

REGISTER_ZEN_FUSED_BATCHNORM_V3_CPU(float, float);
#undef REGISTER_ZEN_FUSED_BATCHNORM_V3_CPU
}  // namespace amd_cpu_plugin
