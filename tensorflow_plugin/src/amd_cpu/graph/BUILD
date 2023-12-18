cc_library(
    name = "cpu_optimizer",
    srcs = ["cpu_optimizer.cc"],
    hdrs = ["cpu_optimizer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin/src/amd_cpu/graph/auto_mixed_precision",
        "//tensorflow_plugin/src/amd_cpu/graph/graph_view",
        "//tensorflow_plugin/src/amd_cpu/graph/remapper",
        "//tensorflow_plugin/src/amd_cpu/graph/zendnn:zen_layout",
    ],
    alwayslink = True,
)

cc_library(
    name = "config_util",
    srcs = ["config_util.cc"],
    hdrs = ["config_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin/src/amd_cpu:protos_all_cc",
    ],
    alwayslink = True,
)
