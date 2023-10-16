cc_library(
    name = "cpu_optimizer",
    srcs = ["cpu_optimizer.cc"],
    hdrs = ["cpu_optimizer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin/src/amd_cpu/graph/zendnn:zen_layout",
        "//tensorflow_plugin/src/amd_cpu/graph/remapper"
    ],
    alwayslink = True,
)
