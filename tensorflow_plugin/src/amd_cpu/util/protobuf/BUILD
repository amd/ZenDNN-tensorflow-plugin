load(
    "//tensorflow_plugin/src/amd_cpu/util:build_config.bzl",
    "tf_proto_library",
    "tf_pyclif_proto_library",
)

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

COMMON_PROTO_SRCS = [
    "config.proto",
]

[
    [
        tf_pyclif_proto_library(
            name = "%s_pyclif" % proto_name,
            proto_lib = ":for_core_protos",
            proto_srcfile = "%s.proto" % proto_name,
            visibility = ["//visibility:public"],
        ),
    ]
    for proto_name in [
        "config",
    ]
]

tf_proto_library(
    name = "for_core_protos",
    srcs = COMMON_PROTO_SRCS,
    cc_api_version = 2,
    make_default_target_header_only = False,
)
