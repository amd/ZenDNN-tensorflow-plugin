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
        "@double_conversion//:double-conversion",
        "@eigen_archive//:eigen",
        "@local_config_tf//:protos_all",
        "@local_config_tf//:tf_header_lib",
        "@nsync//:nsync_cpp",
    ],
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
    ],
)
