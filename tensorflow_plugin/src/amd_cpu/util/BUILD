load("//third_party/zen_dnn:build_defs.bzl", "zendnn_deps")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

filegroup(
    name = "common_util_hdrs",
    srcs = glob(["*.h"]),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "common_utils",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    linkstatic = 1,
    deps = [
        "//tensorflow_plugin/src/amd_cpu/util/gtl:gtl_libs",
        "//third_party/eigen3",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/container:btree",
         # Abseil logging
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:log_sink",
        "@com_google_absl//absl/log:vlog_is_on",
        "@com_google_absl//absl/log:globals",
        "@com_google_absl//absl/utility:if_constexpr",
         # Abseil status
        "@com_google_absl//absl/status:status",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@double_conversion//:double-conversion",
        "@eigen_archive//:eigen",
        "@local_config_tf//:protos_all",
        "@local_config_tf//:tf_header_lib",
        "@nsync//:nsync_cpp",
    ] + zendnn_deps(),
)

cc_library(
    name = "platform",
    hdrs = ["platform.h"],
)

cc_library(
    name = "prefetch",
    hdrs = ["prefetch.h"],
    visibility = ["//visibility:public"],
    deps = [":platform"],
)

cc_library(
    name = "tstring",
    hdrs = [
        "ctstring.h",
        "tstring.h",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":platform",
        ":tstring",
        "@local_config_tf//:protos_all",
        "@local_config_tf//:tf_header_lib",
    ],
)
