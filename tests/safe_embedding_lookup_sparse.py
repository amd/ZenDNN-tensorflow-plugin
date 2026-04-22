# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

embedding_weights = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8]],
  dtype=tf.float32)
sparse_ids = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [2, 0]],
   values=[0, 1, 3, 2], dense_shape=(3, 2))

result = tf.nn.safe_embedding_lookup_sparse(
  embedding_weights,
  sparse_ids,
  sparse_weights=None,
  combiner='mean',
  default_id=None,
  max_norm=None,
  name=None,
  allow_fast_lookup=False
)
print(result.numpy())
