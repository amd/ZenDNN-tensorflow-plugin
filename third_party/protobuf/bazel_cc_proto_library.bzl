# Stub for protobuf v3 compatibility with rules_cc 0.0.17.
# In protobuf v5+, this file provides a Starlark cc_proto_library.
# In protobuf v3, cc_proto_library is a native rule, so we wrap it.

def cc_proto_library(**kwargs):
    native.cc_proto_library(**kwargs)
