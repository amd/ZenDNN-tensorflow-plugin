[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)](https://badge.fury.io/py/zentf)
[![PyPI](https://badge.fury.io/py/tensorflow.svg)](https://badge.fury.io/py/zentf)

**The latest ZenDNN Plugin for TensorFlow*** **(zentf) 5.0 is here!**

This powerful upgrade continues to redefine deep learning performance on AMD EPYC™ CPUs, combining relentless optimization, innovative features, and industry-leading support for modern workloads.

zentf 5.0 takes deep learning to new heights with significant enhancements for bfloat16 performance and expanded support for cutting-edge models like Llama 3.1 and 3.2, Microsoft Phi, amongst others.

Under the hood, zentf introduces robust improvements in kernel performance and efficiency. The zentf plugin leverages microkernels and operators from the ZenDNN 5.0 library. Notable updates to the ZenDNN library include enhancements to BFloat16 and related fusions to better leverage the EPYC microarchitecture and cache hierarchy. The zentf 5.0 plugin-in has been specifically optimized to accelerate the performance of generative LLM models on EPYC. Some of these enhancements are additive in nature -- for e.g., optimized attention or positional embedding operators. The zentf plugin also includes custom pattern matching passes to select the fusions that maximize throughput and minimize latency.

The zentf 5.0 plugs seamlessly with TensorFlow version 2.17, offering a high-performance experience for deep learning on AMD EPYC™ platforms.

## Support

We welcome feedback, suggestions, and bug reports. Should you have any of the these, please kindly file an issue on the ZenDNN Plugin for TensorFlow Github page: https://github.com/amd/ZenDNN-tensorflow-plugin/issues

## License

AMD copyrighted code in ZenDNN is subject to the [Apache-2.0, MIT, or BSD-3-Clause](https://github.com/amd/ZenDNN-tensorflow-plugin/blob/main/LICENSE) licenses; consult the source code file headers for the applicable license. Third party copyrighted code in ZenDNN is subject to the licenses set forth in the source code file headers of such code.
