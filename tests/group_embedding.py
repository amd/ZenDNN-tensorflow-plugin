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

"""
Validation test for _ZenGroupEmbedding fusion.

Builds the graph pattern that the remapper fuses:
 N x (GatherV2 -> Cast(float->bfloat16) -> SafeCast) + ConcatV2
and compares the fused result against a NumPy reference.

Tests axis=0 (standard embedding), axis=1, and axis=-1.
"""

import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


def numpy_reference(table, indices_list, axis):
  """Compute the expected result: gather each, cast to bf16, concat."""
  gathered = []
  for indices in indices_list:
    g = np.take(table, indices, axis=axis)
    g_bf16 = g.astype(np.float32)  # simulate bf16 round-trip
    # SafeCast: clamp inf to zero (matches kernel post-processing)
    g_bf16[np.isinf(g_bf16)] = 0.0
    gathered.append(g_bf16)
  return np.concatenate(gathered, axis=axis)


def build_fused_graph(table_shape, num_indices_list, axis):
  """Build the GatherV2 -> Cast -> SafeCast -> ConcatV2 pattern (single table).

  Uses placeholders to prevent constant folding so the remapper
  can see and fuse the GatherV2 + Cast + ConcatV2 pattern.
  """
  table_ph = tf.compat.v1.placeholder(tf.float32, shape=table_shape,
                    name="table")
  indices_phs = []
  gathers = []
  for i, n_idx in enumerate(num_indices_list):
    idx_ph = tf.compat.v1.placeholder(tf.int64, shape=[n_idx],
                     name=f"indices_{i}")
    indices_phs.append(idx_ph)
    g = tf.raw_ops.GatherV2(params=table_ph, indices=idx_ph, axis=axis)
    g_cast = tf.cast(g, tf.bfloat16)
    is_inf = tf.math.is_inf(g_cast)
    zeros = tf.zeros_like(g_cast)
    g_safe = tf.where(is_inf, zeros, g_cast)
    gathers.append(g_safe)
  concat = tf.concat(gathers, axis=axis)
  result = tf.cast(concat, tf.float32)
  return result, [table_ph], indices_phs


def build_multi_table_graph(table_specs, axis):
  """Build a multi-table GatherV2 -> Cast -> SafeCast -> ConcatV2 pattern.

  table_specs: list of (table_shape, num_indices_list) per table group.
  All gathers feed into one ConcatV2 — triggers full-fusion.
  """
  table_phs = []
  indices_phs = []
  gathers = []
  for t, (t_shape, idx_counts) in enumerate(table_specs):
    t_ph = tf.compat.v1.placeholder(tf.float32, shape=t_shape,
                    name=f"table_{t}")
    table_phs.append(t_ph)
    for i, n_idx in enumerate(idx_counts):
      idx_ph = tf.compat.v1.placeholder(
        tf.int64, shape=[n_idx],
        name=f"indices_t{t}_{i}")
      indices_phs.append(idx_ph)
      g = tf.raw_ops.GatherV2(params=t_ph, indices=idx_ph, axis=axis)
      g_cast = tf.cast(g, tf.bfloat16)
      is_inf = tf.math.is_inf(g_cast)
      zeros = tf.zeros_like(g_cast)
      g_safe = tf.where(is_inf, zeros, g_cast)
      gathers.append(g_safe)
  concat = tf.concat(gathers, axis=axis)
  result = tf.cast(concat, tf.float32)
  return result, table_phs, indices_phs


def _apply_safecast(tensor):
  """Apply one SafeCast layer: SelectV2(IsInf(x), ZerosLike(x), x)."""
  is_inf = tf.math.is_inf(tensor)
  zeros = tf.zeros_like(tensor)
  return tf.where(is_inf, zeros, tensor)


def build_fp32_fused_graph(table_shape, num_indices_list, axis):
  """Build the FP32 pattern: GatherV2 -> SafeCast -> SafeCast -> ConcatV2.

  No Cast — stays float32 throughout with double SafeCast layers.
  """
  table_ph = tf.compat.v1.placeholder(tf.float32, shape=table_shape,
                    name="table")
  indices_phs = []
  gathers = []
  for i, n_idx in enumerate(num_indices_list):
    idx_ph = tf.compat.v1.placeholder(tf.int64, shape=[n_idx],
                     name=f"indices_{i}")
    indices_phs.append(idx_ph)
    g = tf.raw_ops.GatherV2(params=table_ph, indices=idx_ph, axis=axis)
    g = _apply_safecast(g)
    g = _apply_safecast(g)
    gathers.append(g)
  concat = tf.concat(gathers, axis=axis)
  return concat, [table_ph], indices_phs


def build_fp32_multi_table_graph(table_specs, axis):
  """Build FP32 multi-table: GatherV2 -> SafeCast -> SafeCast -> ConcatV2."""
  table_phs = []
  indices_phs = []
  gathers = []
  for t, (t_shape, idx_counts) in enumerate(table_specs):
    t_ph = tf.compat.v1.placeholder(tf.float32, shape=t_shape,
                    name=f"table_{t}")
    table_phs.append(t_ph)
    for i, n_idx in enumerate(idx_counts):
      idx_ph = tf.compat.v1.placeholder(
        tf.int64, shape=[n_idx],
        name=f"indices_t{t}_{i}")
      indices_phs.append(idx_ph)
      g = tf.raw_ops.GatherV2(params=t_ph, indices=idx_ph, axis=axis)
      g = _apply_safecast(g)
      g = _apply_safecast(g)
      gathers.append(g)
  concat = tf.concat(gathers, axis=axis)
  return concat, table_phs, indices_phs


def numpy_fp32_reference(table, indices_list, axis):
  """Compute expected result for FP32 pattern (no type conversion)."""
  gathered = []
  for indices in indices_list:
    g = np.take(table, indices, axis=axis).astype(np.float32)
    g[np.isinf(g)] = 0.0
    gathered.append(g)
  return np.concatenate(gathered, axis=axis)


def run_fp32_test(name, table_np, indices_list_np, axis):
  """Run a single-table FP32 test case."""
  tf.compat.v1.reset_default_graph()

  ref = numpy_fp32_reference(table_np, indices_list_np, axis)

  num_indices_list = [len(idx) for idx in indices_list_np]
  with tf.device("/CPU:0"):
    result_op, table_phs, indices_phs = build_fp32_fused_graph(
      table_np.shape, num_indices_list, axis)

  feed = {table_phs[0]: table_np}
  for ph, idx_np in zip(indices_phs, indices_list_np):
    feed[ph] = idx_np

  config = tf.compat.v1.ConfigProto(allow_soft_placement=False)
  with tf.compat.v1.Session(config=config) as sess:
    result = sess.run(result_op, feed_dict=feed)

  print(f"Test name: {name}")
  if np.allclose(result, ref, atol=1e-6, rtol=1e-5):
    print(f"PASS: {name}  shape={result.shape}")
  else:
    max_diff = np.max(np.abs(result - ref))
    print(f"FAIL: {name}  shape={result.shape}  max_diff={max_diff:.6f}")
    print(f"  expected (first 5): {ref.flat[:5]}")
    print(f"  got      (first 5): {result.flat[:5]}")


def run_fp32_multi_table_test(name, tables_and_indices, axis):
  """Run a multi-table FP32 test case."""
  tf.compat.v1.reset_default_graph()

  ref_parts = []
  for table_np, indices_list in tables_and_indices:
    for indices in indices_list:
      g = np.take(table_np, indices, axis=axis).astype(np.float32)
      g[np.isinf(g)] = 0.0
      ref_parts.append(g)
  ref = np.concatenate(ref_parts, axis=axis)

  table_specs = []
  all_indices_np = []
  for table_np, indices_list in tables_and_indices:
    table_specs.append((table_np.shape,
              [len(idx) for idx in indices_list]))
    all_indices_np.extend(indices_list)

  with tf.device("/CPU:0"):
    result_op, table_phs, indices_phs = build_fp32_multi_table_graph(
      table_specs, axis)

  feed = {}
  for ph, (table_np, _) in zip(table_phs, tables_and_indices):
    feed[ph] = table_np
  for ph, idx_np in zip(indices_phs, all_indices_np):
    feed[ph] = idx_np

  config = tf.compat.v1.ConfigProto(allow_soft_placement=False)
  with tf.compat.v1.Session(config=config) as sess:
    result = sess.run(result_op, feed_dict=feed)

  print(f"Test name: {name}")
  if np.allclose(result, ref, atol=1e-6, rtol=1e-5):
    print(f"PASS: {name}  shape={result.shape}")
  else:
    max_diff = np.max(np.abs(result - ref))
    print(f"FAIL: {name}  shape={result.shape}  max_diff={max_diff:.6f}")
    print(f"  expected (first 5): {ref.flat[:5]}")
    print(f"  got      (first 5): {result.flat[:5]}")


def numpy_multi_table_reference(tables_and_indices, axis,
                passthrough_np=None):
  """Reference for multi-table gather+concat, with optional passthrough."""
  gathered = []
  for table_np, indices_list in tables_and_indices:
    for indices in indices_list:
      g = np.take(table_np, indices, axis=axis)
      g = g.astype(np.float32)
      g[np.isinf(g)] = 0.0
      gathered.append(g)
  if passthrough_np is not None:
    gathered.append(passthrough_np)
  return np.concatenate(gathered, axis=axis)


def run_test(name, table_np, indices_list_np, axis):
  """Run a single-table test case and report pass/fail."""
  tf.compat.v1.reset_default_graph()

  ref = numpy_reference(table_np, indices_list_np, axis)

  num_indices_list = [len(idx) for idx in indices_list_np]
  with tf.device("/CPU:0"):
    result_op, table_phs, indices_phs = build_fused_graph(
      table_np.shape, num_indices_list, axis)

  feed = {table_phs[0]: table_np}
  for ph, idx_np in zip(indices_phs, indices_list_np):
    feed[ph] = idx_np

  config = tf.compat.v1.ConfigProto(allow_soft_placement=False)
  with tf.compat.v1.Session(config=config) as sess:
    result = sess.run(result_op, feed_dict=feed)

  print(f"Test name: {name}")
  if np.allclose(result, ref, atol=0.05, rtol=0.01):
    print(f"PASS: {name}  shape={result.shape}")
  else:
    max_diff = np.max(np.abs(result - ref))
    print(f"FAIL: {name}  shape={result.shape}  max_diff={max_diff:.6f}")
    print(f"  expected (first 5): {ref.flat[:5]}")
    print(f"  got      (first 5): {result.flat[:5]}")


def build_passthrough_graph(table_specs, passthrough_shape, axis):
  """Build multi-table gathers + trailing non-gather tensor -> ConcatV2.

  The trailing tensor simulates the AddV2 pattern in the SNAP model.
  """
  table_phs = []
  indices_phs = []
  gathers = []
  for t, (t_shape, idx_counts) in enumerate(table_specs):
    t_ph = tf.compat.v1.placeholder(tf.float32, shape=t_shape,
                    name=f"table_{t}")
    table_phs.append(t_ph)
    for i, n_idx in enumerate(idx_counts):
      idx_ph = tf.compat.v1.placeholder(
        tf.int64, shape=[n_idx],
        name=f"indices_t{t}_{i}")
      indices_phs.append(idx_ph)
      g = tf.raw_ops.GatherV2(params=t_ph, indices=idx_ph, axis=axis)
      g_cast = tf.cast(g, tf.bfloat16)
      is_inf = tf.math.is_inf(g_cast)
      zeros = tf.zeros_like(g_cast)
      g_safe = tf.where(is_inf, zeros, g_cast)
      gathers.append(g_safe)

  pt_ph = tf.compat.v1.placeholder(tf.bfloat16, shape=passthrough_shape,
                   name="passthrough")
  gathers.append(pt_ph)

  concat = tf.concat(gathers, axis=axis)
  result = tf.cast(concat, tf.float32)
  return result, table_phs, indices_phs, pt_ph


def run_passthrough_test(name, tables_and_indices, passthrough_np, axis):
  """Run a test with multi-table gathers + trailing passthrough."""
  tf.compat.v1.reset_default_graph()

  ref = numpy_multi_table_reference(tables_and_indices, axis,
                   passthrough_np.astype(np.float32))

  table_specs = []
  all_indices_np = []
  for table_np, indices_list in tables_and_indices:
    table_specs.append((table_np.shape,
              [len(idx) for idx in indices_list]))
    all_indices_np.extend(indices_list)

  with tf.device("/CPU:0"):
    result_op, table_phs, indices_phs, pt_ph = build_passthrough_graph(
      table_specs, passthrough_np.shape, axis)

  feed = {}
  for ph, (table_np, _) in zip(table_phs, tables_and_indices):
    feed[ph] = table_np
  for ph, idx_np in zip(indices_phs, all_indices_np):
    feed[ph] = idx_np
  feed[pt_ph] = passthrough_np

  config = tf.compat.v1.ConfigProto(allow_soft_placement=False)
  with tf.compat.v1.Session(config=config) as sess:
    result = sess.run(result_op, feed_dict=feed)

  print(f"Test name: {name}")
  if np.allclose(result, ref, atol=0.05, rtol=0.01):
    print(f"PASS: {name}  shape={result.shape}")
  else:
    max_diff = np.max(np.abs(result - ref))
    print(f"FAIL: {name}  shape={result.shape}  max_diff={max_diff:.6f}")
    print(f"  expected (first 5): {ref.flat[:5]}")
    print(f"  got      (first 5): {result.flat[:5]}")


def run_multi_table_test(name, tables_and_indices, axis):
  """Run a multi-table test case and report pass/fail."""
  tf.compat.v1.reset_default_graph()

  ref = numpy_multi_table_reference(tables_and_indices, axis)

  table_specs = []
  all_indices_np = []
  for table_np, indices_list in tables_and_indices:
    table_specs.append((table_np.shape,
              [len(idx) for idx in indices_list]))
    all_indices_np.extend(indices_list)

  with tf.device("/CPU:0"):
    result_op, table_phs, indices_phs = build_multi_table_graph(
      table_specs, axis)

  feed = {}
  for ph, (table_np, _) in zip(table_phs, tables_and_indices):
    feed[ph] = table_np
  for ph, idx_np in zip(indices_phs, all_indices_np):
    feed[ph] = idx_np

  config = tf.compat.v1.ConfigProto(allow_soft_placement=False)
  with tf.compat.v1.Session(config=config) as sess:
    result = sess.run(result_op, feed_dict=feed)

  print(f"Test name: {name}")
  if np.allclose(result, ref, atol=0.05, rtol=0.01):
    print(f"PASS: {name}  shape={result.shape}")
  else:
    max_diff = np.max(np.abs(result - ref))
    print(f"FAIL: {name}  shape={result.shape}  max_diff={max_diff:.6f}")
    print(f"  expected (first 5): {ref.flat[:5]}")
    print(f"  got      (first 5): {result.flat[:5]}")


def main():
  np.random.seed(42)

  # --- Test 1: axis=0, standard embedding lookup ---
  table_2d = np.random.randn(100, 16).astype(np.float32)
  idx_a = np.array([0, 5, 10, 50, 99], dtype=np.int64)
  idx_b = np.array([1, 2, 3, 4, 5], dtype=np.int64)
  idx_c = np.array([10, 20, 30, 40, 50], dtype=np.int64)
  run_test("axis=0, 3 lookups, table [100,16]",
       table_2d, [idx_a, idx_b, idx_c], axis=0)

  # --- Test 2: axis=1 ---
  table_3d = np.random.randn(4, 20, 8).astype(np.float32)
  idx_d = np.array([0, 3, 7, 19], dtype=np.int64)
  idx_e = np.array([1, 5, 10, 15], dtype=np.int64)
  run_test("axis=1, 2 lookups, table [4,20,8]",
       table_3d, [idx_d, idx_e], axis=1)

  # --- Test 3: axis=-1 (last axis) ---
  table_2d_wide = np.random.randn(3, 50).astype(np.float32)
  idx_f = np.array([0, 10, 25, 49], dtype=np.int64)
  idx_g = np.array([5, 15, 35, 45], dtype=np.int64)
  idx_h = np.array([2, 12, 22, 32], dtype=np.int64)
  run_test("axis=-1, 3 lookups, table [3,50]",
       table_2d_wide, [idx_f, idx_g, idx_h], axis=-1)

  # --- Test 4: axis=0, single index (degenerate) ---
  idx_single = np.array([42], dtype=np.int64)
  run_test("axis=0, 2 lookups, single index",
       table_2d, [idx_single, idx_single], axis=0)

  # --- Test 5: axis=1, larger batch ---
  table_large = np.random.randn(8, 64, 32).astype(np.float32)
  idx_l1 = np.random.randint(0, 64, size=10).astype(np.int64)
  idx_l2 = np.random.randint(0, 64, size=10).astype(np.int64)
  idx_l3 = np.random.randint(0, 64, size=10).astype(np.int64)
  idx_l4 = np.random.randint(0, 64, size=10).astype(np.int64)
  run_test("axis=1, 4 lookups, table [8,64,32]",
       table_large, [idx_l1, idx_l2, idx_l3, idx_l4], axis=1)

  # =================================================================
  # Multi-table tests (full-fusion: different tables in one ConcatV2)
  # =================================================================

  # --- Test 6: 2 different tables, axis=-1, SNAP-like pattern ---
  table_A = np.random.randn(8, 50).astype(np.float32)
  table_B = np.random.randn(8, 10).astype(np.float32)
  idx_m1 = np.array([5], dtype=np.int64)
  idx_m2 = np.array([10], dtype=np.int64)
  idx_m3 = np.array([25], dtype=np.int64)
  idx_m4 = np.array([3], dtype=np.int64)
  idx_m5 = np.array([7], dtype=np.int64)
  run_multi_table_test(
    "multi-table: 2 tables [8,50]+[8,10], axis=-1, 3+2 lookups",
    [
      (table_A, [idx_m1, idx_m2, idx_m3]),
      (table_B, [idx_m4, idx_m5]),
    ],
    axis=-1)

  # --- Test 7: 3 table groups (A, B, A pattern like SNAP) ---
  table_C = np.random.randn(4, 30).astype(np.float32)
  table_D = np.random.randn(4, 8).astype(np.float32)
  idx_n1 = np.array([0], dtype=np.int64)
  idx_n2 = np.array([15], dtype=np.int64)
  idx_n3 = np.array([29], dtype=np.int64)
  idx_n4 = np.array([2], dtype=np.int64)
  idx_n5 = np.array([5], dtype=np.int64)
  idx_n6 = np.array([10], dtype=np.int64)
  run_multi_table_test(
    "multi-table: A-B-A pattern [4,30]+[4,8]+[4,30], axis=-1",
    [
      (table_C, [idx_n1, idx_n2]),
      (table_D, [idx_n4]),
      (table_C, [idx_n5, idx_n6]),
    ],
    axis=-1)

  # --- Test 8: multi-table with batch_size=1 (like SNAP query features) ---
  table_E = np.random.randn(1, 136).astype(np.float32)
  table_F = np.random.randn(1, 6).astype(np.float32)
  idx_p = [np.array([i], dtype=np.int64) for i in range(5)]
  idx_q = [np.array([i], dtype=np.int64) for i in range(3)]
  idx_r = [np.array([i + 50], dtype=np.int64) for i in range(4)]
  run_multi_table_test(
    "multi-table: [1,136]+[1,6]+[1,136], axis=-1, 5+3+4 lookups",
    [
      (table_E, idx_p),
      (table_F, idx_q),
      (table_E, idx_r),
    ],
    axis=-1)

  # =================================================================
  # Passthrough tests (trailing non-gather tensor absorbed into fusion)
  # =================================================================

  # --- Test 9: multi-table gathers + trailing passthrough, axis=-1 ---
  table_G = np.random.randn(8, 50).astype(np.float32)
  table_H = np.random.randn(8, 10).astype(np.float32)
  pt_data = np.random.randn(8, 3).astype(np.float32)
  idx_pt1 = np.array([5], dtype=np.int64)
  idx_pt2 = np.array([10], dtype=np.int64)
  idx_pt3 = np.array([3], dtype=np.int64)
  run_passthrough_test(
    "passthrough: 2 tables + trailing [8,3], axis=-1",
    [
      (table_G, [idx_pt1, idx_pt2]),
      (table_H, [idx_pt3]),
    ],
    pt_data,
    axis=-1)

  # --- Test 10: simpler passthrough test with known values ---
  table_I = np.random.randn(4, 20).astype(np.float32)
  table_J = np.random.randn(4, 5).astype(np.float32)
  pt_simple = np.ones((4, 2), dtype=np.float32)
  idx_s1 = np.array([0], dtype=np.int64)
  idx_s2 = np.array([3], dtype=np.int64)
  idx_s3 = np.array([1], dtype=np.int64)
  idx_s4 = np.array([4], dtype=np.int64)
  # Use float32 passthrough to match T_output when no SafeCast
  run_passthrough_test(
    "passthrough: A-B-A + trailing ones [4,2], axis=-1",
    [
      (table_I, [idx_s1, idx_s2]),
      (table_J, [idx_s3]),
      (table_I, [idx_s4]),
    ],
    pt_simple,
    axis=-1)

  # =================================================================
  # FP32 tests (double SafeCast, no Cast — stays float32 throughout)
  # =================================================================

  # --- Test 11: FP32 single table, axis=-1 ---
  table_fp = np.random.randn(8, 50).astype(np.float32)
  idx_fp1 = np.array([0, 10, 25, 49], dtype=np.int64)
  idx_fp2 = np.array([5, 15, 35, 45], dtype=np.int64)
  idx_fp3 = np.array([2, 12, 22, 32], dtype=np.int64)
  run_fp32_test("FP32: axis=-1, 3 lookups, table [8,50]",
         table_fp, [idx_fp1, idx_fp2, idx_fp3], axis=-1)

  # --- Test 12: FP32 single table, axis=0 ---
  table_fp0 = np.random.randn(100, 16).astype(np.float32)
  idx_fp4 = np.array([0, 5, 10, 50, 99], dtype=np.int64)
  idx_fp5 = np.array([1, 2, 3, 4, 5], dtype=np.int64)
  run_fp32_test("FP32: axis=0, 2 lookups, table [100,16]",
         table_fp0, [idx_fp4, idx_fp5], axis=0)

  # --- Test 13: FP32 multi-table, axis=-1, A-B-A pattern ---
  table_fpA = np.random.randn(4, 30).astype(np.float32)
  table_fpB = np.random.randn(4, 8).astype(np.float32)
  idx_fp6 = np.array([0], dtype=np.int64)
  idx_fp7 = np.array([15], dtype=np.int64)
  idx_fp8 = np.array([2], dtype=np.int64)
  idx_fp9 = np.array([5], dtype=np.int64)
  idx_fp10 = np.array([10], dtype=np.int64)
  run_fp32_multi_table_test(
    "FP32 multi-table: A-B-A [4,30]+[4,8]+[4,30], axis=-1",
    [
      (table_fpA, [idx_fp6, idx_fp7]),
      (table_fpB, [idx_fp8]),
      (table_fpA, [idx_fp9, idx_fp10]),
    ],
    axis=-1)

  print("\nAll tests completed.")


if __name__ == "__main__":
  main()
