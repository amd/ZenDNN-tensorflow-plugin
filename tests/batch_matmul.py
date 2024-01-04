# ******************************************************************************
# Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
tf.compat.v1.disable_eager_execution()
a = tf.random.normal(shape=[1, 24, 24, 8], dtype=tf.float32, seed=5)
b = tf.random.normal(shape=[1, 24, 8, 24], dtype=tf.float32, seed=5)

with tf.device("/CPU:0"):
  c = tf.raw_ops.BatchMatMul(x=a, y=b)

sess = tf.compat.v1.Session(
    config=tf.compat.v1.ConfigProto(
        allow_soft_placement=False,
        log_device_placement=True))
print(sess.run(c))
