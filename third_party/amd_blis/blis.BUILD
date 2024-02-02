genrule(
    name = "blis_build",
    srcs = [
        "configure",
        "build/config.mk.in",
        "build/bli_config.h.in",
        "Makefile",
        "common.mk",
    ],
    outs = [
        "include/zen/bli_config.h",
        "include/zen/blis.h",
        "include/zen/cblas.h",
        "libblis-mt.so.4",
    ],
    cmd = "cd external/amd_blis " +
          "&& make clean " +
          "&& make distclean " +
          "&& ./configure -a aocl_gemm --prefix=$(RULEDIR) --disable-static --enable-threading=openmp --enable-cblas amdzen " +
          "&& make -j install " +
          "&& cd ../.. " +
          "&& mkdir -p $(RULEDIR)/include/zen/ " +
          "&& cp external/amd_blis/bli_config.h $(location include/zen/bli_config.h) " +
          "&& cp external/amd_blis/include/**/blis.h $(location include/zen/blis.h) " +
          "&& cp external/amd_blis/include/**/cblas.h $(location include/zen/cblas.h) " +
          "&& cp external/amd_blis/lib/**/libblis-mt.so.4 $(location  libblis-mt.so.4) ",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "amd_blis",
    srcs = [":blis_build"],
    includes = ["include/zen/"],
    visibility = ["//visibility:public"],
)
