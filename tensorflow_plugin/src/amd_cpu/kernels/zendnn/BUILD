load("//tensorflow_plugin:amd_cpu_plugin.bzl", "tf_copts")
load("//third_party/zen_dnn:build_defs.bzl", "zendnn_deps")

package(licenses = ["notice"])  # Apache 2.0

cc_library(
    name = "fused_eigen_output_kernels",
    srcs = ["fused_eigen_output_kernels.cc"],
    hdrs = ["fused_eigen_output_kernels.h"],
    deps = [
        "//tensorflow_plugin/src/amd_cpu/util:common_utils",
        "//third_party/eigen3",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "zen_kernel_common",
    srcs = ["zen_kernel_common.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = zendnn_deps(),
    alwayslink = True,
)

cc_library(
    name = "zen_softmax_kernel",
    srcs = ["zen_softmax_kernel.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":zen_kernel_common",
        ":zen_mempool",
        "//tensorflow_plugin/src/amd_cpu/util:common_utils",
    ] + zendnn_deps(),
    alwayslink = True,
)

cc_library(
    name = "zen_conv2d_kernel",
    srcs = [
        "zen_conv2d_fused_kernel.cc",
        "zen_conv2d_kernel.cc",
        "zen_conv_kernel.cc",
        "zen_kernel_utils.cc",
    ],
    hdrs = [
        "zen_conv_kernel.h",
        "zen_conv_kernel_fused.h",
        "zen_conv_kernel_util.h",
    ],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":fused_eigen_output_kernels",
        ":zen_kernel_common",
        ":zen_mempool",
        "@com_google_absl//absl/container:inlined_vector",
        "@local_config_tf//:tf_header_lib",
    ] + zendnn_deps(),
    alwayslink = True,
)

cc_library(
    name = "zen_matmul_kernels",
    srcs = ["zen_matmul_kernel.cc"],
    hdrs = ["zen_matmul_kernel_util.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":fused_eigen_output_kernels",
        ":zen_kernel_common",
        ":zen_mempool",
        "//tensorflow_plugin/src/amd_cpu/util:common_utils",
    ] + zendnn_deps(),
    alwayslink = True,
)

cc_library(
    name = "zen_batch_matmul_kernels",
    srcs = ["zen_batch_matmul_kernel.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":zen_kernel_common",
        ":zen_mempool",
        "//tensorflow_plugin/src/amd_cpu/util:common_utils",
    ] + zendnn_deps(),
    alwayslink = True,
)

cc_library(
    name = "zen_pooling_kernels",
    srcs = [
        "pooling_ops_common.cc",
        "zen_pool_kernel.cc",
    ],
    hdrs = ["pooling_ops_common.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":zen_kernel_common",
        ":zen_mempool",
        "//tensorflow_plugin/src/amd_cpu/util:common_utils",
    ] + zendnn_deps(),
    alwayslink = True,
)

cc_library(
    name = "zen_fused_batchnorm_kernel",
    srcs = ["zen_fused_batchnorm_kernel.cc"],
    hdrs = ["zen_fused_batchnorm_kernel.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":fused_eigen_output_kernels",
        ":zen_kernel_common",
        ":zen_mempool",
        "//tensorflow_plugin/src/amd_cpu/util:common_utils",
    ] + zendnn_deps(),
    alwayslink = True,
)

cc_library(
    name = "zen_transpose_kernel",
    srcs = ["zen_transpose_kernel.cc"],
    hdrs = ["zen_transpose_functor.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":zen_kernel_common",
        ":zen_mempool",
        "//tensorflow_plugin/src/amd_cpu/util:common_utils",
    ] + zendnn_deps(),
    alwayslink = True,
)

cc_library(
    name = "zen_reshape_kernel",
    srcs = ["zen_reshape_kernel.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":zen_kernel_common",
        ":zen_mempool",
        "//tensorflow_plugin/src/amd_cpu/util:common_utils",
    ] + zendnn_deps(),
    alwayslink = True,
)

ZEN_KERNELS = [
    ":zen_softmax_kernel",
    ":zen_conv2d_kernel",
    ":zen_matmul_kernels",
    ":zen_pooling_kernels",
    ":zen_fused_batchnorm_kernel",
    ":zen_transpose_kernel",
    ":zen_batch_matmul_kernels",
    ":zen_reshape_kernel",
]

cc_library(
    name = "zen_mempool",
    srcs = ["zen_mempool.cc"],
    hdrs = ["zen_mempool.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin/src/amd_cpu/util:common_utils",
    ] + zendnn_deps(),
    alwayslink = True,
)

cc_library(
    name = "zen_kernels_impl",
    srcs = ["zen_kernels_init.cc"],
    hdrs = ["zen_kernels_init.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@local_config_gcc//:framework_lib",
        "@local_config_tf//:tf_header_lib",
    ] + ZEN_KERNELS,
    alwayslink = True,
)
