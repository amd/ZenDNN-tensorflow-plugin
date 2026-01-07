load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@rules_foreign_cc//foreign_cc:cmake.bzl", "cmake")

package(
    default_visibility = [
        "//tensorflow_plugin:__subpackages__",
    ],
    licenses = ["notice"],
)

exports_files(["LICENSE"])

config_setting(
    name = "build_with_zendnn",
    define_values = {
        "build_with_zendnn": "true",
        "build_with_openmp": "true",
    },
    visibility = ["//visibility:public"],
)

filegroup(
    name = "LICENSES",
    srcs = [
        "LICENSE",
    ],
    visibility = ["//visibility:public"],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
)

filegroup(
    name = "zendnn_allfiles",
    srcs = glob(
       ["**/*"],
       exclude = ["**/.git/**"]
    ),
)

filegroup(
    name = "zendnn_includes",
    srcs = ["inc", "src/common"],
    visibility = ["//visibility:public"],
)

cc_library(
	name = "libamdZenDNN.so",
	srcs = [],
	deps = [":zendnn_cmake_lib"],
    includes = [
        "inc",
        "src/common",
        "src/common/ittnotify",
        "src/cpu",
        "src/cpu/gemm",
        "src/cpu/matmul",
        "src/cpu/rnn",
        "src/cpu/x64",
        "src/cpu/x64/xbyak",
    ],
    visibility = ["//visibility:public"],
)


cmake(
    name = "zendnn_cmake_lib",
    install = True,
    cache_entries = {
       "AMDBLIS_TAG" : "AOCL-Sep2025-b1",
       "BLIS_API" : "1",
       "BUILD_SHARED_LIBS" : "ON",
       "_GLIBCXX_USE_CXX11_ABI" : "1",
       "AMDBLIS_ENABLE_BLAS" : "ON",
       "LPGEMM_V5_0" : "1",
       "FBGEMM_ENABLE" : "0",
       "FBGEMM_BUILD_TESTS" : "0",
    },
    build_args = ["-j$(CPU_COUNT)", ],
    env = {
       "CMAKE_ARGS": "--log-level=WARNING -DCMAKE_VERBOSE_MAKEFILE=OFF",
       "CXXFLAGS": "-Wno-error",
       "CFLAGS": "-Wno-error",
    },
    lib_source = ":zendnn_allfiles",
    out_shared_libs = ["libamdZenDNN.so", "libblis-mt.so.5"],

    postfix_script = """
        #!/bin/bash

        echo "=== ZenDNN Post-build: Copying library to expected location ==="

        # Find the actual library location.
        ACTUAL_LIB=$(find . -name "*.so*" -print)

        if [ -n "$ACTUAL_LIB" ]; then
            echo "Found ZenDNN and BLIS libraries, $ACTUAL_LIB"

            # Create the expected directory structure.
            mkdir -p "$INSTALLDIR/lib"

            # Copy the library to expected location.
            for i in "$ACTUAL_LIB"
            do 
              cp -v $i "$INSTALLDIR/lib" 
            done

            echo "Copied library to: $INSTALLDIR/lib"
            ls -la "$INSTALLDIR/lib/"
        else
            echo "ERROR: Could not find libamdZenDNN.so or libblis-mt.so"
            pwd
            find . -name "*.so*" -type f -print
            exit 1
        fi

        echo "=== ZenDNN Post-build completed successfully ==="
    """,


    visibility = ["//visibility:public"],
)
