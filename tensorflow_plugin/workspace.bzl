load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls", "third_party_http_archive")
load("//third_party/build_option:gcc_configure.bzl", "gcc_configure")
load("//third_party/systemlibs:syslibs_configure.bzl", "syslibs_configure")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def clean_dep(dep):
    return str(Label(dep))

def amd_cpu_plugin_workspace(path_prefix = "", tf_repo_name = ""):
    """All external dependencies for TF builds"""
    gcc_configure(name = "local_config_gcc")
    syslibs_configure(name = "local_config_syslibs")

    # Add rules_foreign_cc for CMake integration
    http_archive(
        name = "rules_foreign_cc",
        sha256 = "4b33d62cf109bcccf286b30ed7121129cc34cf4f4ed9d8a11f38d9108f40ba74",
        strip_prefix = "rules_foreign_cc-0.11.1",
        url = "https://github.com/bazelbuild/rules_foreign_cc/releases/download/0.11.1/rules_foreign_cc-0.11.1.tar.gz",
    )

    tf_http_archive(
        name = "zendnnl_repo",
        build_file = "//third_party:zendnnl_cmake.BUILD",
        sha256 = "54d2c02b4a1cc900b12fa89431dd87b4b870b8c7fbc530cf1a755f26bebf88b7",
        strip_prefix = "ZenDNN-5.2.1",
        urls = tf_mirror_urls("https://github.com/amd/ZenDNN/archive/refs/tags/v5.2.1.tar.gz"),
    )

    http_archive(
        name = "bazel_toolchains",
        sha256 = "294cdd859e57fcaf101d4301978c408c88683fbc46fbc1a3829da92afbea55fb",
        strip_prefix = "bazel-toolchains-8c717f8258cd5f6c7a45b97d974292755852b658",
        urls = tf_mirror_urls("https://github.com/bazelbuild/bazel-toolchains/archive/8c717f8258cd5f6c7a45b97d974292755852b658.tar.gz"),
    )

    # https://github.com/bazelbuild/bazel-skylib/releases
    # Updated to 1.7.1 for TF 2.21 / protobuf v6.31.1 (needs paths.is_normalized)
    http_archive(
        name = "bazel_skylib",
        sha256 = "bc283cdfcd526a52c3201279cda4bc298652efa898b10b4db0837dc51652756f",
        urls = tf_mirror_urls(
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.7.1/bazel-skylib-1.7.1.tar.gz",
        ),
    )

    http_archive(
        name = "rules_pkg",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.7.1/rules_pkg-0.7.1.tar.gz",
            "https://github.com/bazelbuild/rules_pkg/releases/download/0.7.1/rules_pkg-0.7.1.tar.gz",
        ],
        sha256 = "451e08a4d78988c06fa3f9306ec813b836b1d076d0f055595444ba4ff22b867f",
    )

    http_archive(
        name = "rules_proto",
        sha256 = "20b240eba17a36be4b0b22635aca63053913d5c1ee36e16be36499d167a2f533",
        strip_prefix = "rules_proto-11bf7c25e666dd7ddacbcd4d4c4a9de7a25175f8",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_proto/archive/11bf7c25e666dd7ddacbcd4d4c4a9de7a25175f8.tar.gz",
            "https://github.com/bazelbuild/rules_proto/archive/11bf7c25e666dd7ddacbcd4d4c4a9de7a25175f8.tar.gz",
        ],
    )

    tf_http_archive(
        name = "rules_cc",
        sha256 = "abc605dd850f813bb37004b77db20106a19311a96b2da1c92b789da529d28fe1",
        strip_prefix = "rules_cc-0.0.17",
        urls = tf_mirror_urls("https://github.com/bazelbuild/rules_cc/releases/download/0.0.17/rules_cc-0.0.17.tar.gz"),
    )

    # Protobuf v6.31.1 requires rules_java 8.6.1 (see protobuf_deps.bzl)
    # rules_java 8.6.1 archive has no top-level directory
    tf_http_archive(
        name = "rules_java",
        sha256 = "c5bc17e17bb62290b1fd8fdd847a2396d3459f337a7e07da7769b869b488ec26",
        urls = tf_mirror_urls("https://github.com/bazelbuild/rules_java/releases/download/8.6.1/rules_java-8.6.1.tar.gz"),
    )

    # Abseil LTS 20250814.1 (TF 2.21 version)
    SYS_DIRS = [
        "algorithm",
        "base",
        "cleanup",
        "container",
        "debugging",
        "flags",
        "functional",
        "hash",
        "memory",
        "meta",
        "numeric",
        "random",
        "status",
        "strings",
        "synchronization",
        "time",
        "types",
        "utility",
    ]
    SYS_LINKS = {
        "//third_party/absl:system.absl.{name}.BUILD".format(name = n): "absl/{name}/BUILD.bazel".format(name = n)
        for n in SYS_DIRS
    }

    tf_http_archive(
        name = "com_google_absl",
        sha256 = "d1abe9da2003e6cbbd7619b0ced3e52047422f4f4ac6c66a9bef5d2e99fea837",
        build_file = clean_dep("//third_party/absl:com_google_absl.BUILD"),
        system_build_file = clean_dep("//third_party/absl:system.BUILD"),
        system_link_files = SYS_LINKS,
        strip_prefix = "abseil-cpp-d38452e1ee03523a208362186fd42248ff2609f6",
        urls = tf_mirror_urls("https://github.com/abseil/abseil-cpp/archive/d38452e1ee03523a208362186fd42248ff2609f6.tar.gz"),
    )

    http_archive(
        name = "bazel_features",
        sha256 = "4fd9922d464686820ffd8fcefa28ccffa147f7cdc6b6ac0d8b07fde565c65d66",
        strip_prefix = "bazel_features-1.25.0",
        urls = [
            "https://mirror.bazel.build/github.com/bazel-contrib/bazel_features/releases/download/v1.25.0/bazel_features-v1.25.0.tar.gz",
            "https://github.com/bazel-contrib/bazel_features/releases/download/v1.25.0/bazel_features-v1.25.0.tar.gz",
        ],
    )

    tf_http_archive(
        name = "eigen_archive",
        build_file = clean_dep("//third_party:eigen.BUILD"),
        sha256 = "a71517b3815984c1a8174db1ebc58a17d4f5c23c06e377bbc4a5dfc85855a516",  # SHARED_EIGEN_SHA
        strip_prefix = "eigen-dcbaf2d608f306450f1e74949eb87e9a22a7ef4b",
        urls = tf_mirror_urls("https://gitlab.com/libeigen/eigen/-/archive/dcbaf2d608f306450f1e74949eb87e9a22a7ef4b/eigen-dcbaf2d608f306450f1e74949eb87e9a22a7ef4b.tar.gz"),
    )

    tf_http_archive(
        name = "double_conversion",
        build_file = clean_dep("//third_party:double_conversion.BUILD"),
        sha256 = "2f7fbffac0d98d201ad0586f686034371a6d152ca67508ab611adc2386ad30de",
        strip_prefix = "double-conversion-3992066a95b823efc8ccc1baf82a1cfc73f6e9b8",
        system_build_file = clean_dep("//third_party/systemlibs:double_conversion.BUILD"),
        urls = tf_mirror_urls("https://github.com/google/double-conversion/archive/3992066a95b823efc8ccc1baf82a1cfc73f6e9b8.zip"),
    )

    tf_http_archive(
        name = "zlib",
        build_file = clean_dep("//third_party:zlib.BUILD"),
        sha256 = "9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23",
        strip_prefix = "zlib-1.3.1",
        system_build_file = clean_dep("//third_party/systemlibs:zlib.BUILD"),
        urls = tf_mirror_urls("https://zlib.net/zlib-1.3.1.tar.gz"),
    )

    # Use http_archive (not tf_http_archive) to enable repo_mapping
    # repo_mapping ensures protobuf's @abseil-cpp references resolve to @com_google_absl
    # This prevents dual abseil repos and undeclared inclusion errors
    http_archive(
        name = "com_google_protobuf",
        sha256 = "6e09bbc950ba60c3a7b30280210cd285af8d7d8ed5e0a6ed101c72aff22e8d88",
        strip_prefix = "protobuf-6.31.1",
        urls = tf_mirror_urls("https://github.com/protocolbuffers/protobuf/archive/refs/tags/v6.31.1.zip"),
        repo_mapping = {
            "@abseil-cpp": "@com_google_absl",
        },
    )

    tf_http_archive(
        name = "rules_python",
        sha256 = "d70cd72a7a4880f0000a6346253414825c19cdd40a28289bdf67b8e6480edff8",
        strip_prefix = "rules_python-0.28.0",
        urls = tf_mirror_urls("https://github.com/bazelbuild/rules_python/releases/download/0.28.0/rules_python-0.28.0.tar.gz"),
    )

    tf_http_archive(
        name = "nsync",
        sha256 = "caf32e6b3d478b78cff6c2ba009c3400f8251f646804bcb65465666a9cea93c4",
        strip_prefix = "nsync-1.22.0",
        system_build_file = clean_dep("//third_party/systemlibs:nsync.BUILD"),
        urls = tf_mirror_urls("https://github.com/google/nsync/archive/1.22.0.tar.gz"),
    )

    # Intel openMP that is part of LLVM sources.
    tf_http_archive(
        name = "llvm_openmp",
        build_file = "//third_party/llvm_openmp:BUILD",
        sha256 = "d19f728c8e04fb1e94566c8d76aef50ec926cd2f95ef3bf1e0a5de4909b28b44",
        strip_prefix = "openmp-10.0.1.src",
        urls = tf_mirror_urls("https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.1/openmp-10.0.1.src.tar.xz"),
    )
