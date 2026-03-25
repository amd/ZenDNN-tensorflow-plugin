# Platform-specific build configurations.
# TF 2.16-2.19: Uses proto_gen from protobuf.bzl (protobuf v3.21.9)

load("@com_google_protobuf//:protobuf.bzl", "proto_gen")
load("//tensorflow_plugin:workspace.bzl", "clean_dep")
load("@rules_cc//cc:defs.bzl", "cc_library")

def cc_proto(name, src, deps = []):
    native.genrule(
        name = "%s_cc" % name,
        outs = ["include/protos/%s.pb.cc" % name, "include/protos/%s.pb.h" % name],
        cmd = "$(location @com_google_protobuf//:protoc) -I$(GENDIR)/external/local_config_tf/include/protos --cpp_out=$(GENDIR)/external/local_config_tf/include/protos $<",
        srcs = ["include/protos/%s" % src],
        tools = ["@com_google_protobuf//:protoc"],
    )
    native.cc_library(
        name = "%s_proto" % name,
        srcs = ["include/protos/%s.pb.cc" % name],
        hdrs = ["include/protos/%s.pb.h" % name],
        deps = [
            "@com_google_protobuf//:protobuf_headers",
            "@com_google_protobuf//:protobuf",
        ] + deps,
        copts = ["-I$(GENDIR)/external/local_config_tf/include/protos"],
    )

def if_static(extra_deps = [], otherwise = []):
    return otherwise

def if_not_windows(extra_deps = [], otherwise = []):
    return extra_deps

def well_known_proto_libs():
    """Set of standard protobuf protos, like Any and Timestamp.

    This list should be provided by protobuf.bzl, but it's not.
    """
    return [
        "@com_google_protobuf//:any_proto",
        "@com_google_protobuf//:api_proto",
        "@com_google_protobuf//:compiler_plugin_proto",
        "@com_google_protobuf//:descriptor_proto",
        "@com_google_protobuf//:duration_proto",
        "@com_google_protobuf//:empty_proto",
        "@com_google_protobuf//:field_mask_proto",
        "@com_google_protobuf//:source_context_proto",
        "@com_google_protobuf//:struct_proto",
        "@com_google_protobuf//:timestamp_proto",
        "@com_google_protobuf//:type_proto",
        "@com_google_protobuf//:wrappers_proto",
    ]

def tf_deps(deps, suffix):
    tf_deps = []

    for dep in deps:
        tf_dep = dep

        if not ":" in dep:
            dep_pieces = dep.split("/")
            tf_dep += ":" + dep_pieces[len(dep_pieces) - 1]

        tf_deps += [tf_dep + suffix]

    return tf_deps

def pyx_library(
        name,
        deps = [],
        py_deps = [],
        srcs = [],
        testonly = None,
        srcs_version = "PY2AND3",
        **kwargs):
    py_srcs = []
    pyx_srcs = []
    pxd_srcs = []
    for src in srcs:
        if src.endswith(".pyx") or (src.endswith(".py") and
                                    src[:-3] + ".pxd" in srcs):
            pyx_srcs.append(src)
        elif src.endswith(".py"):
            py_srcs.append(src)
        else:
            pxd_srcs.append(src)
        if src.endswith("__init__.py"):
            pxd_srcs.append(src)

    for filename in pyx_srcs:
        native.genrule(
            name = filename + "_cython_translation",
            srcs = [filename],
            outs = [filename.split(".")[0] + ".cpp"],
            cmd = "PYTHONHASHSEED=0 $(location @cython//:cython_binary) --cplus $(SRCS) --output-file $(OUTS)",
            testonly = testonly,
            tools = ["@cython//:cython_binary"] + pxd_srcs,
        )

    shared_objects = []
    for src in pyx_srcs:
        stem = src.split(".")[0]
        shared_object_name = stem + ".so"
        native.cc_binary(
            name = shared_object_name,
            srcs = [stem + ".cpp"],
            deps = deps,
            linkshared = 1,
            testonly = testonly,
        )
        shared_objects.append(shared_object_name)

    native.py_library(
        name = name,
        srcs = py_srcs,
        deps = py_deps,
        srcs_version = srcs_version,
        data = shared_objects,
        testonly = testonly,
        **kwargs
    )

def _proto_cc_hdrs(srcs, use_grpc_plugin = False):
    ret = [s[:-len(".proto")] + ".pb.h" for s in srcs]
    if use_grpc_plugin:
        ret += [s[:-len(".proto")] + ".grpc.pb.h" for s in srcs]
    return ret

def _proto_cc_srcs(srcs, use_grpc_plugin = False):
    ret = [s[:-len(".proto")] + ".pb.cc" for s in srcs]
    if use_grpc_plugin:
        ret += [s[:-len(".proto")] + ".grpc.pb.cc" for s in srcs]
    return ret

def _proto_py_outs(srcs, use_grpc_plugin = False):
    ret = [s[:-len(".proto")] + "_pb2.py" for s in srcs]
    if use_grpc_plugin:
        ret += [s[:-len(".proto")] + "_pb2_grpc.py" for s in srcs]
    return ret

def cc_proto_library(
        name,
        srcs = [],
        deps = [],
        cc_libs = [],
        include = None,
        protoc = "@com_google_protobuf//:protoc",
        internal_bootstrap_hack = False,
        use_grpc_plugin = False,
        use_grpc_namespace = False,
        make_default_target_header_only = False,
        protolib_name = None,
        protolib_deps = [],
        **kargs):
    wkt_deps = ["@com_google_protobuf//:cc_wkt_protos"]
    all_protolib_deps = protolib_deps + wkt_deps

    includes = []
    if include != None:
        includes = [include]
    if protolib_name == None:
        protolib_name = name

    if internal_bootstrap_hack:
        proto_gen(
            name = protolib_name + "_genproto",
            srcs = srcs,
            includes = includes,
            protoc = protoc,
            visibility = ["//visibility:public"],
            deps = [s + "_genproto" for s in all_protolib_deps],
        )

        native.cc_library(
            name = name,
            **kargs
        )
        return

    grpc_cpp_plugin = None
    plugin_options = []
    if use_grpc_plugin:
        grpc_cpp_plugin = "//external:grpc_cpp_plugin"
        if use_grpc_namespace:
            plugin_options = ["services_namespace=grpc"]

    gen_srcs = _proto_cc_srcs(srcs, use_grpc_plugin)
    gen_hdrs = _proto_cc_hdrs(srcs, use_grpc_plugin)
    outs = gen_srcs + gen_hdrs

    proto_gen(
        name = protolib_name + "_genproto",
        srcs = srcs,
        outs = outs,
        gen_cc = 1,
        includes = includes,
        plugin = grpc_cpp_plugin,
        plugin_language = "grpc",
        plugin_options = plugin_options,
        protoc = protoc,
        visibility = ["//visibility:public"],
        deps = [s + "_genproto" for s in all_protolib_deps],
    )

    if use_grpc_plugin:
        cc_libs += select({
            clean_dep("//tensorflow:linux_s390x"): ["//external:grpc_lib_unsecure"],
            "//conditions:default": ["//external:grpc_lib"],
        })

    impl_name = name + "_impl"
    header_only_name = name + "_headers_only"
    header_only_deps = tf_deps(protolib_deps, "_cc_headers_only")

    if make_default_target_header_only:
        native.alias(
            name = name,
            actual = header_only_name,
            visibility = kargs["visibility"],
        )
    else:
        native.alias(
            name = name,
            actual = impl_name,
            visibility = kargs["visibility"],
        )

    native.cc_library(
        name = impl_name,
        srcs = gen_srcs,
        hdrs = gen_hdrs,
        deps = cc_libs + deps,
        includes = includes,
        alwayslink = 1,
        **kargs
    )
    native.cc_library(
        name = header_only_name,
        deps = [
            "@com_google_protobuf//:protobuf_headers",
        ] + header_only_deps + if_static([impl_name]),
        hdrs = gen_hdrs,
        **kargs
    )

def py_proto_library(
        name,
        srcs = [],
        deps = [],
        py_libs = [],
        py_extra_srcs = [],
        include = None,
        default_runtime = "@com_google_protobuf//:protobuf_python",
        protoc = "@com_google_protobuf//:protoc",
        use_grpc_plugin = False,
        **kargs):
    outs = _proto_py_outs(srcs, use_grpc_plugin)

    includes = []
    if include != None:
        includes = [include]

    grpc_python_plugin = None
    if use_grpc_plugin:
        grpc_python_plugin = "//external:grpc_python_plugin"

    proto_gen(
        name = name + "_genproto",
        srcs = srcs,
        outs = outs,
        gen_py = 1,
        includes = includes,
        plugin = grpc_python_plugin,
        plugin_language = "grpc",
        protoc = protoc,
        visibility = ["//visibility:public"],
        deps = [s + "_genproto" for s in deps],
    )

    if default_runtime and not default_runtime in py_libs + deps:
        py_libs = py_libs + [default_runtime]

    native.py_library(
        name = name,
        srcs = outs + py_extra_srcs,
        deps = py_libs + deps,
        imports = includes,
        **kargs
    )

def tf_proto_library_cc(
        name,
        srcs = [],
        has_services = None,
        protodeps = [],
        visibility = None,
        testonly = 0,
        cc_libs = [],
        cc_stubby_versions = None,
        cc_grpc_version = None,
        use_grpc_namespace = False,
        j2objc_api_version = 1,
        cc_api_version = 2,
        js_codegen = "jspb",
        create_service = False,
        create_java_proto = False,
        make_default_target_header_only = False):
    js_codegen = js_codegen  # unused argument
    native.filegroup(
        name = name + "_proto_srcs",
        srcs = srcs + tf_deps(protodeps, "_proto_srcs"),
        testonly = testonly,
        visibility = visibility,
    )
    _ignore = (create_service, create_java_proto)

    use_grpc_plugin = None
    if cc_grpc_version:
        use_grpc_plugin = True

    protolib_deps = tf_deps(protodeps, "")
    cc_deps = tf_deps(protodeps, "_cc")
    cc_name = name + "_cc"
    if not srcs:
        proto_gen(
            name = name + "_genproto",
            protoc = "@com_google_protobuf//:protoc",
            visibility = ["//visibility:public"],
            deps = [s + "_genproto" for s in protolib_deps],
        )

        native.alias(
            name = cc_name + "_genproto",
            actual = name + "_genproto",
            testonly = testonly,
            visibility = visibility,
        )

        native.alias(
            name = cc_name + "_headers_only",
            actual = cc_name,
            testonly = testonly,
            visibility = visibility,
        )

        native.cc_library(
            name = cc_name,
            deps = cc_deps + ["@com_google_protobuf//:protobuf_headers"] + if_static([name + "_cc_impl"]),
            testonly = testonly,
            visibility = visibility,
        )
        native.cc_library(
            name = cc_name + "_impl",
            deps = [s + "_impl" for s in cc_deps] + ["@com_google_protobuf//:cc_wkt_protos"],
        )

        return

    cc_proto_library(
        name = cc_name,
        protolib_name = name,
        testonly = testonly,
        srcs = srcs,
        cc_libs = cc_libs + if_static(
            ["@com_google_protobuf//:protobuf"],
            ["@com_google_protobuf//:protobuf_headers"],
        ),
        copts = if_not_windows([
            "-Wno-unknown-warning-option",
            "-Wno-unused-but-set-variable",
            "-Wno-sign-compare",
        ]),
        make_default_target_header_only = make_default_target_header_only,
        protoc = "@com_google_protobuf//:protoc",
        use_grpc_plugin = use_grpc_plugin,
        use_grpc_namespace = use_grpc_namespace,
        visibility = visibility,
        deps = cc_deps + ["@com_google_protobuf//:cc_wkt_protos"],
        protolib_deps = protolib_deps,
    )

def tf_proto_library_py(
        name,
        srcs = [],
        protodeps = [],
        deps = [],
        visibility = None,
        testonly = 0,
        srcs_version = "PY2AND3",
        use_grpc_plugin = False):
    py_deps = tf_deps(protodeps, "_py")
    py_name = name + "_py"
    if not srcs:
        proto_gen(
            name = py_name + "_genproto",
            protoc = "@com_google_protobuf//:protoc",
            visibility = ["//visibility:public"],
            deps = [s + "_genproto" for s in py_deps],
        )
        native.py_library(
            name = py_name,
            deps = py_deps + [clean_dep("@com_google_protobuf//:protobuf_python")],
            testonly = testonly,
            visibility = visibility,
        )
        return

    py_proto_library(
        name = py_name,
        testonly = testonly,
        srcs = srcs,
        default_runtime = clean_dep("@com_google_protobuf//:protobuf_python"),
        protoc = "@com_google_protobuf//:protoc",
        srcs_version = srcs_version,
        use_grpc_plugin = use_grpc_plugin,
        visibility = visibility,
        deps = deps + py_deps + [clean_dep("@com_google_protobuf//:protobuf_python")],
    )

def tf_jspb_proto_library(**kwargs):
    pass

def tf_proto_library(
        name,
        srcs = [],
        has_services = None,
        protodeps = [],
        visibility = None,
        testonly = 0,
        cc_libs = [],
        cc_api_version = 2,
        cc_grpc_version = None,
        use_grpc_namespace = False,
        j2objc_api_version = 1,
        js_codegen = "jspb",
        create_service = False,
        create_java_proto = False,
        make_default_target_header_only = False,
        exports = []):
    """Make a proto library, possibly depending on other proto libraries."""
    _ignore = (js_codegen, exports, create_service, create_java_proto)

    native.proto_library(
        name = name,
        srcs = srcs,
        deps = protodeps + well_known_proto_libs(),
        visibility = visibility,
        testonly = testonly,
    )

    tf_proto_library_cc(
        name = name,
        testonly = testonly,
        srcs = srcs,
        cc_grpc_version = cc_grpc_version,
        use_grpc_namespace = use_grpc_namespace,
        cc_libs = cc_libs,
        make_default_target_header_only = make_default_target_header_only,
        protodeps = protodeps,
        visibility = visibility,
    )

    tf_proto_library_py(
        name = name,
        testonly = testonly,
        srcs = srcs,
        protodeps = protodeps,
        srcs_version = "PY2AND3",
        use_grpc_plugin = has_services,
        visibility = visibility,
    )

def tf_additional_env_hdrs():
    return []

def tf_additional_device_tracer_srcs():
    return ["device_tracer.cc"]

def tf_additional_test_deps():
    return []

def tf_kernel_tests_linkstatic():
    return 0

def tf_additional_lib_deps():
    """Additional dependencies needed to build TF libraries."""
    return [
        "@com_google_absl//absl/base:base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:optional",
    ] + if_static(
        [clean_dep("@nsync//:nsync_cpp")],
        [clean_dep("@nsync//:nsync_headers")],
    )

def tf_py_clif_cc(name, visibility = None, **kwargs):
    pass

def tf_pyclif_proto_library(
        name,
        proto_lib,
        proto_srcfile = "",
        visibility = None,
        **kwargs):
    native.filegroup(name = name)
    native.filegroup(name = name + "_pb2")

def tf_additional_rpc_deps():
    return []

def tf_additional_tensor_coding_deps():
    return []

def tf_fingerprint_deps():
    return [
        "@farmhash_archive//:farmhash",
    ]

def tf_protobuf_deps():
    return if_static(
        [
            clean_dep("@com_google_protobuf//:protobuf"),
        ],
        otherwise = [clean_dep("@com_google_protobuf//:protobuf_headers")],
    )

def tf_protobuf_compiler_deps():
    return if_static(
        [
            clean_dep("@com_google_protobuf//:protobuf"),
        ],
        otherwise = [clean_dep("@com_google_protobuf//:protobuf_headers")],
    )

def tf_portable_deps_no_runtime():
    return [
        "//third_party/eigen3",
        "@double_conversion//:double-conversion",
        "@nsync//:nsync_cpp",
        "@com_googlesource_code_re2//:re2",
        "@farmhash_archive//:farmhash",
    ]

def tf_google_mobile_srcs_no_runtime():
    return []

def tf_google_mobile_srcs_only_runtime():
    return []

def if_llvm_aarch64_available(then, otherwise = []):
    return otherwise

def if_llvm_system_z_available(then, otherwise = []):
    return select({
        "//tensorflow:linux_s390x": then,
        "//conditions:default": otherwise,
    })

def tf_generate_proto_text_sources(name, srcs_relative_dir, srcs, protodeps = [], deps = [], visibility = None, compatible_with = None):
    out_hdrs = (
        [
            p.replace(".proto", ".pb_text.h")
            for p in srcs
        ] + [p.replace(".proto", ".pb_text-impl.h") for p in srcs]
    )
    out_srcs = [p.replace(".proto", ".pb_text.cc") for p in srcs]
    native.genrule(
        name = name + "_srcs",
        srcs = srcs + protodeps + [clean_dep("//tensorflow_plugin/tools/proto_text:placeholder.txt")],
        outs = out_hdrs + out_srcs,
        visibility = visibility,
        cmd =
            "$(location //tensorflow_plugin/tools/proto_text:gen_proto_text_functions) " +
            "$(@D) " + srcs_relative_dir + " $(SRCS)",
        tools = [
            clean_dep("//tensorflow_plugin/tools/proto_text:gen_proto_text_functions"),
        ],
        compatible_with = compatible_with,
    )

    native.filegroup(
        name = name + "_hdrs",
        srcs = out_hdrs,
        visibility = visibility,
        compatible_with = compatible_with,
    )

    cc_library(
        compatible_with = compatible_with,
        name = name,
        srcs = out_srcs,
        hdrs = out_hdrs,
        visibility = visibility,
        deps = deps,
        alwayslink = 1,
    )
