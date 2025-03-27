The ZenDNN plugin for TensorFlow is called zentf. zentf 5.0.2 is a minor upgrade to zentf 5.0 release.

This upgrade includes support for a Java interface to zentf through TensorFlow-Java.

zentf 5.0.2 includes enhancements for bfloat16 performance, primarily by leveraging microkernels and operators from the ZenDNN 5.0.2 library. These operators are designed to better leverage the EPYC microarchitecture and cache hierarchy.

The zentf 5.0.2 plugin works seamlessly with TensorFlow versions from 2.18 to 2.16, offering a high-performance experience for deep learning on AMD EPYC™ platforms.

## Support

We welcome feedback, suggestions, and bug reports. Should you have any of the these, please kindly file an issue on the ZenDNN Plugin for TensorFlow Github page [here](https://github.com/amd/ZenDNN-tensorflow-plugin/issues)

## License

AMD copyrighted code in ZenDNN is subject to the [Apache-2.0, MIT, or BSD-3-Clause](https://github.com/amd/ZenDNN-tensorflow-plugin/blob/main/LICENSE) licenses; consult the source code file headers for the applicable license. Third party copyrighted code in ZenDNN is subject to the licenses set forth in the source code file headers of such code.
