package(default_visibility = ["//visibility:public"])

load(
    "@org_tensorflow_plugin//third_party:common.bzl",
    "template_rule",
)

# This is the main library that provides all the necessary headers from the
# TensorFlow pip package, including the pre-generated .pb.h files.
cc_library(
    name = "tf_header_lib",
    hdrs = [":tf_header_include"],
    includes = ["include"],
    visibility = ["//visibility:public"],
)

# The "protos_all" target is what other parts of the build depend on for proto
# headers.
alias(
    name = "protos_all",
    actual = ":tf_header_lib",
    visibility = ["//visibility:public"],
)

%{TF_HEADER_GENRULE}
%{TF_SHARED_LIBRARY_GENRULE}
