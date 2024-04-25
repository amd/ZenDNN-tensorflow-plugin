[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)](https://badge.fury.io/py/zentf)
[![PyPI](https://badge.fury.io/py/tensorflow.svg)](https://badge.fury.io/py/zentf)

**EARLY ACCESS:** The ZenDNN TensorFlow* Plugin (zenTF) extends TensorFlow* with an innovative upgrade that's set to revolutionize performance on AMD hardware.

As of version 4.2, AMD is unveiling a game-changing upgrade to ZenDNN, introducing a cutting-edge plug-in mechanism and an enhanced architecture under the hood. This isn't just about extensions; ZenDNN's aggressive AMD-specific optimizations operate at every level. It delves into comprehensive graph optimizations, including pattern identification, graph reordering, and seeking opportunities for graph fusions. At the operator level, ZenDNN boasts enhancements with microkernels, mempool optimizations, and efficient multi-threading on the large number of AMD EPYC cores. Microkernel optimizations further exploit all possible low-level math libraries, including AOCL BLIS.

The result? Enhanced performance with respect to baseline TensorFlow*. The ZenDNN TensorFlow* Plugin is compatible with TensorFlow versions 2.16 and later.

## Support

Please note that zenTF is currently in “Early Access” mode. We welcome feedback, suggestions, and bug reports. Should you have any of these, please contact us on zendnn.maintainers@amd.com

## License

AMD copyrighted code in ZenDNN is subject to the [Apache-2.0, MIT, or BSD-3-Clause](https://github.com/amd/ZenDNN-tensorflow-plugin/blob/main/LICENSE) licenses; consult the source code file headers for the applicable license. Third party copyrighted code in ZenDNN is subject to the licenses set forth in the source code file headers of such code.
