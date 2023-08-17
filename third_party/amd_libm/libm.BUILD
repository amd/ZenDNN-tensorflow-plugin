genrule(
    name = "libm_config",
    outs = ["libalm.so"],
    cmd = "cd external/amd_libm" +
          "&& scons -j32" +
          "&& cd ../.." +
          "&& cp external/amd_libm/build/aocl-release/src/libalm.so $(location libalm.so) ",
    visibility = ["//visibility:public"],
)
