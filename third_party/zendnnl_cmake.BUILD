# BUILD file for building ZenDNNL using cmake rule.
load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

# Export all source files for the cmake rule.
# Exclude build artifacts, git files, and other non-source files to improve caching.
filegroup(
    name = "all_srcs",
    srcs = glob(
        ["**/*"],
        exclude = [
            # Build artifacts
            "**/build/**",
            "**/BUILD*",
            "**/CMakeCache.txt",
            "**/CMakeFiles/**",
            "**/*.o",
            "**/*.a",
            "**/*.so",
            "**/*.dylib",
            "**/*.dll",
            "**/*.exe",
            "**/.cmake/**",
            "**/install/**",
            # Version control
            "**/.git/**",
            "**/.gitignore",
            "**/.gitattributes",
            # IDE files
            "**/.vscode/**",
            "**/.idea/**",
            "**/*.swp",
            "**/*.swo",
            "**/*~",
            # Temporary files
            "**/*.tmp",
            "**/*.log",
            "**/.DS_Store",
            "**/Thumbs.db",
        ],
    ),
    visibility = ["//visibility:public"],
)

# Build ZenDNNL using cmake rule.
cmake(
    name = "zendnnl_lib",
    # Use Ninja generator for faster builds and better caching
    # Ninja is faster than Make and handles parallel builds better
    generate_args = ["-GNinja"],
    # Enable parallel builds - Ninja will auto-detect CPU cores by default
    build_args = [
        # Ninja auto-detects cores if no -j flag is provided
        # Uncomment the line below to explicitly set parallel jobs:
        # "-j$(CPU_COUNT)",  # Use CPU_COUNT from .bazelrc
    ],
    install = True,
    cache_entries = {
        # ZenDNNL Configurations.
        "ZENDNNL_DEPENDS_ONEDNN": "ON",
        "ZENDNNL_DEPENDS_AMDBLIS": "OFF",  # Disabled: mutually exclusive with AOCLDLP
        "ZENDNNL_DEPENDS_AOCLUTILS": "ON",
        "ZENDNNL_DEPENDS_JSON": "ON",
        "ZENDNNL_DEPENDS_AOCLDLP": "ON",  # Enabled: Deep Learning Performance library
        "ZENDNNL_DEPENDS_LIBXSMM": "ON",
        "ZENDNNL_DEPENDS_FBGEMM": "ON",
        # ZenDNNL Library Build Options.
        "ZENDNNL_LIB_BUILD_ARCHIVE": "ON",
        "ZENDNNL_LIB_BUILD_SHARED": "OFF",

        # ZenDNNL Optional Components (disable for faster builds).
        "ZENDNNL_BUILD_EXAMPLES": "OFF",
        "ZENDNNL_BUILD_GTEST": "OFF",
        "ZENDNNL_BUILD_DOXYGEN": "OFF",
        "ZENDNNL_BUILD_BENCHDNN": "OFF",
        "ZENDNNL_CODE_COVERAGE": "OFF",

        # CMake Build Configuration.
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_VERBOSE_MAKEFILE": "OFF",
        "CMAKE_CXX_STANDARD": "17",
        "CMAKE_CXX_EXTENSIONS": "OFF",
        "CMAKE_CXX_STANDARD_REQUIRED": "ON",
        # Use Ninja generator for faster builds
        "CMAKE_GENERATOR": "Ninja",
        # Improve caching by using deterministic build paths
        "CMAKE_EXPORT_COMPILE_COMMANDS": "OFF",  # Disable to avoid non-deterministic outputs

        # Override CMAKE_INSTALL_PREFIX.
        "CMAKE_INSTALL_PREFIX": "$INSTALLDIR",

        # Dependency Injection (let ZenDNNL build its own dependencies)
        "ZENDNNL_ONEDNN_INJECTED": "OFF",
        "ZENDNNL_AMDBLIS_INJECTED": "OFF",
        "ZENDNNL_AOCLUTILS_INJECTED": "OFF",
        "ZENDNNL_AOCLDLP_INJECTED": "OFF",
    },
    env = {
        "CXXFLAGS": "-fopenmp -std=c++17",
        "CFLAGS": "-fopenmp",
        "LDFLAGS": "-lgomp",
        "CMAKE_C_COMPILER": "gcc",
        "CMAKE_CXX_COMPILER": "g++",
        "CC": "gcc",
        "CXX": "g++",
        # Ensure deterministic builds
        "SOURCE_DATE_EPOCH": "0",
    },
    lib_source = ":all_srcs",

    # Add out_include_dir to expose headers.
    out_include_dir = "zendnnl/include",

    # Specify the expected output location.
    out_lib_dir = "zendnnl/lib",

    # Expose ALL the static libraries.
    out_static_libs = [
        "libzendnnl_archive.a",
        "libaoclutils.a",
        "libau_cpuid.a",
        "libdnnl.a",
        "libxsmm.a",
        "libaocl-dlp.a",
        "libfbgemm.a",
        "libasmjit.a",
        "libcpuinfo.a",
    ],

    # Copy all libraries and headers to the right locations.
    postfix_script = """
        echo "=== ZenDNNL Post-build: Copying libraries and headers ==="

        # Set up variables.
        BUILD_DIR="$PWD"
        INSTALL_BASE="$BUILD_DIR/install"

        # Create target directory structure once.
        mkdir -p "$INSTALLDIR/zendnnl/lib" "$INSTALLDIR/zendnnl/include"

        # Helper function for REQUIRED libraries.
        copy_lib() {
            local lib_path="$1"
            local lib_name="$2"
            local display_name="$3"

            if [ -f "$lib_path" ]; then
                cp "$lib_path" "$INSTALLDIR/zendnnl/lib/$lib_name"
                echo "$display_name copied"
                return 0
            else
                echo "ERROR: $display_name not found at $lib_path"
                exit 1
            fi
        }

        # Helper function for REQUIRED headers.
        copy_headers() {
            local header_path="$1"
            local display_name="$2"

            if [ -d "$header_path" ]; then
                cp -r "$header_path"/* "$INSTALLDIR/zendnnl/include/"
                echo "$display_name headers copied"
                return 0
            else
                echo "ERROR: $display_name headers not found at $header_path"
                exit 1
            fi
        }

        # Copy main ZenDNNL library.
        copy_lib "$INSTALL_BASE/zendnnl/lib/libzendnnl_archive.a" "libzendnnl_archive.a" "ZenDNNL library"

        # Copy ZenDNNL headers.
        copy_headers "$INSTALL_BASE/zendnnl/include" "ZenDNNL"

        # Copy JSON headers.
        copy_headers "$INSTALL_BASE/deps/json/include" "JSON"

        # Copy AOCL Utils headers.
        copy_headers "$INSTALL_BASE/deps/aoclutils/include" "AOCL Utils"

        # Copy AOCL DLP headers.
        copy_headers "$INSTALL_BASE/deps/aocldlp/include" "AOCL DLP"

        # Copy LIBXSMM headers.
        copy_headers "$INSTALL_BASE/deps/libxsmm/include" "LIBXSMM"

        # Copy FBGEMM headers.
        copy_headers "$INSTALL_BASE/deps/fbgemm/include" "FBGEMM"

        # Copy OneDNN headers.
        copy_headers "$INSTALL_BASE/deps/onednn/include" "OneDNN"

        # TODO(plugin): Improve and simplify the following copy operation.
        # Copy operators directory with full structure (includes matmul/aocl_dlp/).
        OPERATORS_SRC="$INSTALL_BASE/zendnnl/include/operators"
        if [ -d "$OPERATORS_SRC" ]; then
            mkdir -p "$INSTALLDIR/zendnnl/include/operators"
            cp -r "$OPERATORS_SRC"/* "$INSTALLDIR/zendnnl/include/operators/"
            echo "Operators headers copied (preserving directory structure)"
        else
            echo "ERROR: Operators headers not found at $OPERATORS_SRC"
            exit 1
        fi

        # WORKAROUND: Copy lowoha_common.hpp from source tree if missing
        # (CMake FILE_SET bug - file is declared as public but not installed)
        LOWOHA_COMMON_INSTALLED="$INSTALLDIR/zendnnl/include/lowoha_operators/matmul/lowoha_common.hpp"
        if [ ! -f "$LOWOHA_COMMON_INSTALLED" ]; then
            LOWOHA_COMMON_SRC="$EXT_BUILD_ROOT/external/zendnnl_repo/zendnnl/src/lowoha_operators/matmul/lowoha_common.hpp"
            if [ -f "$LOWOHA_COMMON_SRC" ]; then
                cp "$LOWOHA_COMMON_SRC" "$LOWOHA_COMMON_INSTALLED"
                echo "lowoha_common.hpp copied from source (CMake install workaround)"
            else
                echo "ERROR: lowoha_common.hpp not found in source at $LOWOHA_COMMON_SRC"
                exit 1
            fi
        fi

        # === REQUIRED DEPENDENCY LIBRARIES ===

        echo "DEBUG: ZENDNNL_MANYLINUX_BUILD=${ZENDNNL_MANYLINUX_BUILD:-unset}"

        # Determine library directory based on MANYLINUX build setting
        if [ "${ZENDNNL_MANYLINUX_BUILD:-}" = "1" ] || [ "${ZENDNNL_MANYLINUX_BUILD:-}" = "true" ]; then
            LIB_DIR="lib64"
        else
            LIB_DIR="lib"
        fi

        # Copy AOCL Utils libraries.
        copy_lib "$INSTALL_BASE/deps/aoclutils/$LIB_DIR/libaoclutils.a" "libaoclutils.a" "AOCL Utils library"
        copy_lib "$INSTALL_BASE/deps/aoclutils/$LIB_DIR/libau_cpuid.a" "libau_cpuid.a" "AOCL CPUID library"

        # Copy OneDNN library.
        copy_lib "$INSTALL_BASE/deps/onednn/$LIB_DIR/libdnnl.a" "libdnnl.a" "OneDNN library"

        # Copy AOCL DLP library.
        copy_lib "$INSTALL_BASE/deps/aocldlp/lib/libaocl-dlp.a" "libaocl-dlp.a" "AOCL DLP library"

        # Copy LIBXSMM library.
        copy_lib "$INSTALL_BASE/deps/libxsmm/lib/libxsmm.a" "libxsmm.a" "LIBXSMM library"

        # Copy FBGEMM library.
        copy_lib "$INSTALL_BASE/deps/fbgemm/$LIB_DIR/libfbgemm.a" "libfbgemm.a" "FBGEMM library"

        # Copy asmjit library (FBGEMM dependency).
        copy_lib "$INSTALL_BASE/deps/fbgemm/$LIB_DIR/libasmjit.a" "libasmjit.a" "asmjit library"

        # Copy cpuinfo library (FBGEMM dependency).
        copy_lib "$INSTALL_BASE/deps/fbgemm/$LIB_DIR/libcpuinfo.a" "libcpuinfo.a" "cpuinfo library"

        echo "=== ZenDNNL Post-build completed successfully ==="
    """,

    visibility = ["//visibility:public"],
)
