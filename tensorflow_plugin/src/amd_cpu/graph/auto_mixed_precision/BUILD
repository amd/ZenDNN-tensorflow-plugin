cc_library(
    name = "auto_mixed_precision",
    srcs = ["auto_mixed_precision.cc"],
    hdrs = [
        "auto_mixed_precision.h",
        "auto_mixed_precision_lists.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin/src/amd_cpu/graph/graph_view:mutable_graph_view",
        "//tensorflow_plugin/src/amd_cpu/graph:config_util",
        "//tensorflow_plugin/src/amd_cpu/graph/utils:graph_properties",
        "//tensorflow_plugin/src/amd_cpu/graph/utils:grappler_item",
        "//tensorflow_plugin/src/amd_cpu/graph/utils:node_type_attr_map",
        "//tensorflow_plugin/src/amd_cpu/graph/utils:symbolic_shapes",
        "//tensorflow_plugin/src/amd_cpu/util:common_utils",
        "//tensorflow_plugin/src/amd_cpu:protos_all_cc",
    ],
    alwayslink = True,
)
