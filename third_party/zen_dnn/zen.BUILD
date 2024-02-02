exports_files(["LICENSE"])

_COPTS_LIST = select({
    "//conditions:default": ["-fexceptions -fopenmp -march=znver2"],
}) + [
    "-DBIAS_ENABLED=1",
    "-DZENDNN_ENABLE=1",
    "-DZENDNN_X64=1",
] + ["-Iexternal/amd_blis/include/zen/"]

_INCLUDES_LIST = [
    "inc",
    "include",
    "src",
    "src/common",
    "src/common/ittnotify",
    "src/cpu",
    "src/cpu/gemm",
    "src/cpu/x64/xbyak",
]

_TEXTUAL_HDRS_LIST = glob([
    "inc/*",
    "include/**/*",
    "src/common/*.hpp",
    "src/common/ittnotify/**/*.h",
    "src/cpu/*.hpp",
    "src/cpu/**/*.hpp",
    "src/cpu/jit_utils/**/*.hpp",
    "src/cpu/x64/xbyak/*.h",
])

cc_binary(
    name = "libamdZenDNN.so",
    srcs = glob(
        [
            "src/common/*.cpp",
            "src/cpu/*.cpp",
            "src/cpu/**/*.cpp",
        ],
        exclude = [
            "src/cpu/aarch64/**",
            "src/common/ittnotify/*.c",
        ],
    ),
    copts = _COPTS_LIST,
    includes = _INCLUDES_LIST,
    linkopts = ["-lm -lpthread -lrt"],
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = ["@amd_blis"],
)
