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

import numpy as np
import sys

def results_comparison(res_file1, res_file2):
  # For BF16 results, observing difference in third precision for some
  # elements hence keeping the rtol to be 0.01
  res1 = np.load(res_file1)
  res2 = np.load(res_file2)
  results_diff = np.isclose(res1, res2, atol=0.05, rtol=0.01)
  if (results_diff.all()):
    print ("onednn and ZenDNN results are matching")
  else:
    print ("ERROR: onednn and ZenDNN results are not matching")

res_file1 = sys.argv[1]
res_file2 = sys.argv[2]
results_comparison(res_file1, res_file2)
