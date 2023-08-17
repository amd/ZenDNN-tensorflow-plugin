def if_zendnn(if_true, if_false = []):
    """Shorthand for select()'ing on whether we're building with ZenDNN.

      ZenDNN gets built if we are building on platforms that support ZenDNN
      (x86 linux) or if specifcially configured to use ZenDNN.

    Args:
      if_true: expression to evaluate if building with ZenDNN.
      if_false: expression to evaluate if building without ZenDNN.

    Returns:
      a select evaluating to either if_true or if_false as appropriate.

    """
    return select({
        "@org_tensorflow_plugin//third_party/zen_dnn:build_with_zendnn": if_true,
        "//conditions:default": if_false,
    })

def zendnn_deps():
    """Returns the correct set of ZenDNN library dependencies.

      Shorthand for select() to pull in the correct set of ZenDNN library deps
      depending on the platform. x86 Linux/Windows with or without
      --config=zendnn will always build with ZenDNN library.

    Returns:
      a select evaluating to a list of library dependencies, suitable for
      inclusion in the deps attribute of rules.
    """
    return select({
        "@org_tensorflow_plugin//third_party/zen_dnn:build_with_zendnn": ["@zen_dnn//:libamdZenDNN.so"],
        "//conditions:default": [],
    })
