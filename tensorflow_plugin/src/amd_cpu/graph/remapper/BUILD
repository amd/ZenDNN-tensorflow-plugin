cc_library(
    name = "remapper",
    srcs = ["remapper.cc"],
    hdrs = [
        "constant_names.h",
        "remapper.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin/src/amd_cpu/graph/utils:graph_common_utils",
        "//tensorflow_plugin/src/amd_cpu/graph/utils:graph_properties",
        "//tensorflow_plugin/src/amd_cpu/graph/utils:graph_view",
        "//tensorflow_plugin/src/amd_cpu/graph/utils:grappler_item",
        "//tensorflow_plugin/src/amd_cpu/graph/utils:layout_utils",
        "//tensorflow_plugin/src/amd_cpu/graph/utils:pattern_utils",
        "//tensorflow_plugin/src/amd_cpu/graph/utils:symbolic_shapes",
    ],
    alwayslink = True,
)
