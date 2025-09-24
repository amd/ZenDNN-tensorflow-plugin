load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls", "third_party_http_archive")
load("//third_party/build_option:gcc_configure.bzl", "gcc_configure")
load("//third_party/systemlibs:syslibs_configure.bzl", "syslibs_configure")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//third_party/absl:workspace.bzl", absl = "repo")

def clean_dep(dep):
    return str(Label(dep))

def amd_cpu_plugin_workspace(path_prefix = "", tf_repo_name = ""):
    """All external dependencies for TF builds"""
    gcc_configure(name = "local_config_gcc")
    syslibs_configure(name = "local_config_syslibs")
    absl()

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
        urls = tf_mirror_urls("https://github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz"),
    )

    http_archive(
        name = "rules_pkg",
        urls = tf_mirror_urls("https://github.com/bazelbuild/rules_pkg/releases/download/0.7.1/rules_pkg-0.7.1.tar.gz"),
        sha256 = "451e08a4d78988c06fa3f9306ec813b836b1d076d0f055595444ba4ff22b867f",
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
        sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
        strip_prefix = "zlib-1.2.11",
        system_build_file = clean_dep("//third_party/systemlibs:zlib.BUILD"),
        urls = tf_mirror_urls("https://zlib.net/zlib-1.2.11.tar.gz"),
    )

    tf_http_archive(
        name = "com_google_protobuf",
        patch_file = clean_dep("//third_party/protobuf:protobuf.patch"),
        sha256 = "f66073dee0bc159157b0bd7f502d7d1ee0bc76b3c1eac9836927511bdc4b3fc1",
        strip_prefix = "protobuf-3.21.9",
        system_build_file = clean_dep("//third_party/systemlibs:protobuf.BUILD"),
        system_link_files = {
            "//third_party/systemlibs:protobuf.bzl": "protobuf.bzl",
            "//third_party/systemlibs:protobuf_deps.bzl": "protobuf_deps.bzl",
        },
        urls = tf_mirror_urls("https://github.com/protocolbuffers/protobuf/archive/v3.21.9.zip"),
    )

    tf_http_archive(
        name = "rules_python",
        sha256 = "aa96a691d3a8177f3215b14b0edc9641787abaaa30363a080165d06ab65e1161",
        urls = tf_mirror_urls("https://github.com/bazelbuild/rules_python/releases/download/0.0.1/rules_python-0.0.1.tar.gz"),
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
        sha256 = "622f0da3e5f1af153d5aba7e81e98e6a45898d85127231bfc48848128cbf3c15",
        strip_prefix = "ZenDNN-zendnn-2025-WW38",
        urls = tf_mirror_urls("https://github.com/amd/ZenDNN/archive/refs/tags/zendnn-2025-WW38.tar.gz"),
    )
