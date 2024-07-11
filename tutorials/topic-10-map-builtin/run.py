#!/usr/bin/env cs_python

# Copyright 2024 Cerebras Systems.
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


import argparse
import json
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder # pylint: disable=no-name-in-module

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='the test name')
parser.add_argument('--cmaddr', help='IP:port for CS system')
args = parser.parse_args()
dirname = args.name

# Parse the compile metadata
with open(f"{dirname}/out.json", encoding="utf-8") as json_file:
  compile_data = json.load(json_file)
params = compile_data["params"]
size = int(params["size"])
print(f"size = {size}")

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT

runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

sym_weight = runner.get_id("weight")
sym_sqrt_diag_A = runner.get_id("sqrt_diag_A")
sym_A = runner.get_id("A")
sym_sum = runner.get_id("sum")

runner.load()
runner.run()

A = np.array([[42.0, 42.0, 42.0, 42.0, 42.0],
              [42.0, 42.0, 42.0, 42.0, 42.0],
              [42.0, 42.0, 42.0, 42.0, 42.0],
              [42.0, 42.0, 42.0, 42.0, 42.0],
              [42.0, 42.0, 42.0, 42.0, 42.0]]).astype(np.float32)
B = np.array([10, 20, 30, 40, 50]).astype(np.int32)

def transformation(value: np.array, coeff1: float, coeff2: float, weight: np.array):
  return np.multiply(value, coeff1 + weight) + np.multiply(value, coeff2 + weight)

def reduction(array):
  return sum(array)

np.random.seed(seed=7)

print("step 1: copy weights to device")
weights = np.random.random(size).astype(np.float32)
runner.memcpy_h2d(sym_weight, weights, 0, 0, 1, 1, size, \
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)

print("step 2: call f_run to test @map")
runner.launch("f_run", nonblock=False)

print("step 3: copy results back to host")
sqrt_result = np.zeros(size, np.float32)
runner.memcpy_d2h(sqrt_result, sym_sqrt_diag_A, 0, 0, 1, 1, size, \
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)

sum_result = np.zeros(1, np.int32)
runner.memcpy_d2h(sum_result, sym_sum, 0, 0, 1, 1, 1, \
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)

A_trans_result = np.zeros(size*size, np.float32)
runner.memcpy_d2h(A_trans_result, sym_A, 0, 0, 1, 1, size*size, \
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)

runner.stop()

# Sqrt example
sqrt_expected = np.sqrt(np.diag(A))
np.testing.assert_equal(sqrt_result, sqrt_expected)

# Transformation example
trans_expected = transformation(np.diag(A), 2.0, 6.0, weights)
np.fill_diagonal(A, trans_expected)
np.testing.assert_equal(A_trans_result.reshape((5, 5)), A)

# Reduction example
sum_expected = np.array([reduction(B)], dtype=np.int32)
np.testing.assert_equal(sum_expected, sum_result)

print("SUCCESS!")
