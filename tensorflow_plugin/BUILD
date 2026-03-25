cc_binary(
    name = "libamdcpu_plugin.so",
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = [
        ":zen_libs_linux",
        "//tensorflow_plugin/src/amd_cpu:amd_cpu_opkernels",
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
        "@llvm_openmp//:libiomp5.so",
    ],
    deps = [
        "@zendnnl_repo//:zendnnl_lib",
    ],
    copts = ["-std=c++17", "-fopenmp"],
    linkopts = ["-fopenmp"],
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

config_setting(
    name = "tf_version_2_21",
    define_values = {"tf_main_version": "2.21"},
    visibility = ["//visibility:public"],
)
