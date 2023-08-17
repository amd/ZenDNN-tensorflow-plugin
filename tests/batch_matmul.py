#*******************************************************************************
# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#*******************************************************************************

import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()
a = tf.constant(np.arange(1, 25, dtype=np.float32), shape=[2, 2, 2, 3])
b = tf.constant(np.arange(25, 49, dtype=np.float32), shape=[2, 2, 3, 2])

with tf.device("/CPU:0"):
    c = tf.raw_ops.BatchMatMul(x=a, y=b)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=False, log_device_placement=True))
print(sess.run(c))
