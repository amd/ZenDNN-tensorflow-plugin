#!/usr/bin/env python
# coding=utf-8

# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ******************************************************************************

import tensorflow as tf
import os
import sys
import numpy as np
from tensorflow.core.protobuf import rewriter_config_pb2

tf.compat.v1.disable_eager_execution()

np.random.seed(0)
input_arr = np.random.normal(3, 2.5, size=(1,448,224))
bias_arr = np.random.normal(3, 2.5, size=(448))

def model(graph_options=None):
  a = tf.compat.v1.placeholder(tf.float32, shape=[1, 448, 224])
  b = tf.compat.v1.placeholder(tf.float32, shape=[448])
  shape = tf.reshape(a,(448,224))
  mul = tf.matmul(shape, shape, transpose_b=True)
  bias = tf.nn.bias_add(mul, b)
  reshape = tf.reshape(bias,(1,448,448))
  out = tf.reshape(tf.nn.gelu(reshape), (448,448))
  mul_1 = tf.matmul(out, out, transpose_b=True)
  bias_1 = tf.nn.bias_add(mul_1, b)
  out_1 = tf.reshape(bias_1, (1,448,448))
  sess = tf.compat.v1.Session(
      config=tf.compat.v1.ConfigProto(
          graph_options=graph_options))
  return (sess.run(out_1, feed_dict={a:input_arr,b:bias_arr}))

bf16 = os.getenv("TF_ZENDNN_PLUGIN_BF16")
is_zendnn = os.getenv("TF_ENABLE_ZENDNN_OPTS")
if (bf16 is not None and int(bf16) == 1 and int(is_zendnn) == 0):
  print ("AMP BF16 ENABLED")
  graph_options = tf.compat.v1.GraphOptions(
      rewrite_options=rewriter_config_pb2.RewriterConfig(
        auto_mixed_precision_onednn_bfloat16=
        rewriter_config_pb2.RewriterConfig.ON))
  result = model(graph_options)
else:
  result = model()

res_file = sys.argv[1]
np.save(res_file,result)
