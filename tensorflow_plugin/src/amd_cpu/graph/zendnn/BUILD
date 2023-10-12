load(
    "//tensorflow_plugin/src/amd_cpu/util:build_config.bzl",
    "tf_protobuf_deps",
)

cc_library(
    name = "zen_layout",
    srcs = ["zen_layout.cc"],
    hdrs = ["zen_layout.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_plugin/src/amd_cpu/graph/utils:function",
        "//tensorflow_plugin/src/amd_cpu/graph/utils:graph_properties",
        "//tensorflow_plugin/src/amd_cpu/graph/utils:graph_view",
        "//tensorflow_plugin/src/amd_cpu/graph/utils:grappler_item",
        "//tensorflow_plugin/src/amd_cpu/graph/utils:layout_utils",
        "//tensorflow_plugin/src/amd_cpu/graph/utils:node_type_attr_map",
        "//tensorflow_plugin/src/amd_cpu/util:common_utils",
    ] + tf_protobuf_deps(),
    alwayslink = True,
)
