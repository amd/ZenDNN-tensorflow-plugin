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


import numpy as np
import tensorflow as tf

def run_safe_embedding_lookup_sparse(weights, ids, comb='mean'):
  """Run TF's safe_embedding_lookup_sparse and return numpy result."""
  return tf.nn.safe_embedding_lookup_sparse(
    weights,
    ids,
    sparse_weights=None,
    combiner=comb,
    default_id=None,
    max_norm=None,
  ).numpy()

def test_combiner(comb, weights, ids):
  """Test a single combiner and print pass/fail."""
  res = run_safe_embedding_lookup_sparse(weights, ids, comb=comb)
  print(f"\ncombiner='{comb}':")
  print(f"  shape: {res.shape}")
  print(f"  output:\n{res}")
  return res

# ---------- Test Case 1: Small example ----------
print("=" * 60)
print("Test Case 1: Small embedding table (4x2)")
print("=" * 60)

embedding_weights = tf.constant(
  [[1, 2], [3, 4], [5, 6], [7, 8]], dtype=tf.float32)
sparse_ids = tf.SparseTensor(
  indices=[[0, 0], [0, 1], [1, 0], [2, 0]],
  values=[0, 1, 3, 2],
  dense_shape=(3, 2))

# Expected results:
# sum:   row0=[1+3, 2+4]=[4,6], row1=[7,8], row2=[5,6]
# mean:  row0=[2,3], row1=[7,8], row2=[5,6]
# sqrtn: row0=[4/sqrt(2), 6/sqrt(2)]=[2.828, 4.243], row1=[7,8], row2=[5,6]

results = {}
for combiner in ['sum', 'mean', 'sqrtn']:
  results[combiner] = test_combiner(combiner, embedding_weights, sparse_ids)

# Verify expected values
assert np.allclose(results['sum'][0], [4, 6]), "sum row0 mismatch"
assert np.allclose(results['mean'][0], [2, 3]), "mean row0 mismatch"
assert np.allclose(results['sqrtn'][0], [4/np.sqrt(2), 6/np.sqrt(2)]), \
  "sqrtn row0 mismatch"

# ---------- Test Case 2: Empty rows ----------
print("\n" + "=" * 60)
print("Test Case 2: Sparse input with empty rows")
print("=" * 60)

embedding_weights2 = tf.constant(
  [[10, 20], [30, 40], [50, 60]], dtype=tf.float32)
# Row 1 is empty (no entries), should be zeros in output
sparse_ids2 = tf.SparseTensor(
  indices=[[0, 0], [2, 0], [2, 1]],
  values=[0, 1, 2],
  dense_shape=(3, 2))

for combiner in ['sum', 'mean', 'sqrtn']:
  result = test_combiner(combiner, embedding_weights2, sparse_ids2)
  # Row 1 must be all zeros (empty row)
  assert np.allclose(result[1], [0, 0]), \
    f"Empty row should be zeros for combiner={combiner}, got {result[1]}"

# ---------- Test Case 3: Larger random test ----------
print("\n" + "=" * 60)
print("Test Case 3: Larger random embedding (100x16)")
print("=" * 60)

np.random.seed(42)
VOCAB_SIZE = 100
EMBED_DIM = 16
BATCH_SIZE = 8

embedding_weights3 = tf.constant(
  np.random.randn(VOCAB_SIZE, EMBED_DIM).astype(np.float32))

# Generate random sparse input
NNZ = 20
row_indices = np.sort(np.random.randint(0, BATCH_SIZE, size=NNZ))
col_indices = np.zeros(NNZ, dtype=np.int64)
values = np.random.randint(0, VOCAB_SIZE, size=NNZ)

indices = np.stack([row_indices, col_indices], axis=1)
sparse_ids3 = tf.SparseTensor(
  indices=indices.tolist(),
  values=values.tolist(),
  dense_shape=(BATCH_SIZE, 1))

for combiner in ['sum', 'mean', 'sqrtn']:
  result = test_combiner(combiner, embedding_weights3, sparse_ids3)
  assert result.shape == (BATCH_SIZE, EMBED_DIM), \
    f"Shape mismatch for combiner={combiner}"

# ---------- Test Case 4: Negative values (absorbed filter) ----------
print("\n" + "=" * 60)
print("Test Case 4: Negative sparse values (GreaterEqual filter)")
print("=" * 60)

embedding_weights4 = tf.constant(
  [[10, 20], [30, 40], [50, 60], [70, 80]], dtype=tf.float32)
# Values include negatives (-1, -2) which should be filtered out
# Only values >= 0 are valid embedding indices
sparse_ids4 = tf.SparseTensor(
  indices=[[0, 0], [0, 1], [0, 2], [1, 0], [2, 0], [2, 1]],
  values=[0, -1, 1, -2, 2, 3],
  dense_shape=(3, 3))

for combiner in ['sum', 'mean', 'sqrtn']:
  result = test_combiner(combiner, embedding_weights4, sparse_ids4)
  if combiner == 'sum':
    # row0: indices 0,1 valid (skip -1) -> [10+30, 20+40] = [40, 60]
    # row1: no valid indices (skip -2) -> [0, 0]
    # row2: indices 2,3 valid -> [50+70, 60+80] = [120, 140]
    assert np.allclose(result[0], [40, 60]), \
      f"sum row0 mismatch: {result[0]}"
    assert np.allclose(result[1], [0, 0]), \
      f"sum row1 (all-negative) should be zero: {result[1]}"
    assert np.allclose(result[2], [120, 140]), \
      f"sum row2 mismatch: {result[2]}"
  elif combiner == 'mean':
    # row0: 2 valid entries -> [40/2, 60/2] = [20, 30]
    assert np.allclose(result[0], [20, 30]), \
      f"mean row0 mismatch: {result[0]}"
  elif combiner == 'sqrtn':
    # row0: 2 valid entries -> [40/sqrt(2), 60/sqrt(2)]
    assert np.allclose(result[0], [40/np.sqrt(2), 60/np.sqrt(2)]), \
      f"sqrtn row0 mismatch: {result[0]}"

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
