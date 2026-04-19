load("@rules_cc//cc:defs.bzl", "cc_library", "cc_binary")

cc_library(
    name = "regex_lib",
    srcs = ["regex_lib.cc"],
    hdrs = ["regex_lib.h"],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "regex_test",
    srcs = ["regex_test.cc"],
    deps = [":regex_lib"],
)
