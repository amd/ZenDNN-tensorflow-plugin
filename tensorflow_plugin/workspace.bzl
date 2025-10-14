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

    http_archive(
        name = "bazel_toolchains",
        sha256 = "294cdd859e57fcaf101d4301978c408c88683fbc46fbc1a3829da92afbea55fb",
        strip_prefix = "bazel-toolchains-8c717f8258cd5f6c7a45b97d974292755852b658",
        urls = tf_mirror_urls("https://github.com/bazelbuild/bazel-toolchains/archive/8c717f8258cd5f6c7a45b97d974292755852b658.tar.gz"),
    )

    # https://github.com/bazelbuild/bazel-skylib/releases
    http_archive(
        name = "bazel_skylib",
        sha256 = "74d544d96f4a5bb630d465ca8bbcfe231e3594e5aae57e1edbf17a6eb3ca2506",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
        ],
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

    # rules_java must come before com_google_protobuf for compatibility proxy resolution.
    tf_http_archive(
        name = "rules_java",
        sha256 = "7eb8f5cc499d14a57a33f445deb17615905554a5e5a4d4cfbe8d86e019f26ee1",
        strip_prefix = "rules_java-5.3.5",
        urls = tf_mirror_urls("https://github.com/bazelbuild/rules_java/archive/5.3.5.tar.gz"),
    )

    http_archive(
        name = "com_google_absl",
        sha256 = "16242f394245627e508ec6bb296b433c90f8d914f73b9c026fddb905e27276e8",
        strip_prefix = "abseil-cpp-20250127.0",
        urls = [
            "https://mirror.bazel.build/github.com/abseil/abseil-cpp/archive/refs/tags/20250127.0.tar.gz",
            "https://github.com/abseil/abseil-cpp/archive/refs/tags/20250127.0.tar.gz",
        ],
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
        sha256 = "df23a89e4cdfa7de2d81ee28190bd194413e47ff177c94076f845b32d7280344",  # SHARED_EIGEN_SHA
        strip_prefix = "eigen-5dc2fbabeee17fe023c38756ebde0c1d56472913",
        urls = tf_mirror_urls("https://gitlab.com/libeigen/eigen/-/archive/5dc2fbabeee17fe023c38756ebde0c1d56472913/eigen-5dc2fbabeee17fe023c38756ebde0c1d56472913.tar.gz"),
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

    tf_http_archive(
        name = "com_google_protobuf",
        sha256 = "f645e6e42745ce922ca5388b1883ca583bafe4366cc74cf35c3c9299005136e2",
        strip_prefix = "protobuf-5.28.3",
        system_build_file = clean_dep("//third_party/systemlibs:protobuf.BUILD"),
        system_link_files = {
            "//third_party/systemlibs:protobuf.bzl": "protobuf.bzl",
            "//third_party/systemlibs:protobuf_deps.bzl": "protobuf_deps.bzl",
        },
        urls = tf_mirror_urls("https://github.com/protocolbuffers/protobuf/archive/v5.28.3.zip"),
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

    tf_http_archive(
        name = "amd_blis",
        build_file = "//third_party/amd_blis:blis.BUILD",
        sha256 = "9d639f1a874492ac7668cba2c25623eb6dca7203a4aeae757ce7e62ca87bc319",
        strip_prefix = "blis-AOCL-Sep2025-b1",
        urls = tf_mirror_urls("https://github.com/amd/blis/archive/refs/tags/AOCL-Sep2025-b1.tar.gz"),
    )

    tf_http_archive(
        name = "zen_dnn",
        build_file = "//third_party/zen_dnn:zen.BUILD",
        sha256 = "1d5b6ab252f5aaba9d4a639ffc8033260da6d723ccd37f57219ce958c735b0be",
        strip_prefix = "ZenDNN-zendnn-2025-WW41",
        urls = tf_mirror_urls("https://github.com/amd/ZenDNN/archive/refs/tags/zendnn-2025-WW41.tar.gz"),
    )
