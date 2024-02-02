cc_binary(
    name = "libamdcpu_plugin.so",
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = [
        ":zen_libs_linux",
        "//tensorflow_plugin/src/amd_cpu:amd_cpu_opkernels",
        "@local_config_tf//:_pywrap_tensorflow_internal",
    ],
)

cc_binary(
    name = "libamdcpu_plugin_cc.so",
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = [
        ":zen_libs_linux",
        "//tensorflow_plugin/src/amd_cpu:amd_cpu_opkernels",
    ],
)

cc_library(
    name = "zen_libs_linux",
    srcs = [
        "@amd_blis//:libblis-mt.so.4",
        "@llvm_openmp//:libiomp5.so",
        "@zen_dnn//:libamdZenDNN.so",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_x86_64",
    values = {"cpu": "k8"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "tf_version_2_12",
    define_values = {"tf_main_version": "2.12"},
    visibility = ["//visibility:public"],
)
