def blis_deps():
    """Returns the correct set of blis library dependencies.

      Shorthand for select() to pull in the correct set of BLIS library deps

    Returns:
      a select evaluating to a list of library dependencies, suitable for
      inclusion in the deps attribute of rules.
    """
    return select({
        "@org_tensorflow_plugin//third_party/zen_dnn:build_with_zendnn": ["@amd_blis//:amd_blis"],
        "//conditions:default": [],
    })
