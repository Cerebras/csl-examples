#!/usr/bin/env cs_python

# Copyright 2023 Cerebras Systems.
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

from cerebras.sdk.debug.debug_util import debug_util
from cerebras.sdk.sdk_utils import memcpy_view, input_array_to_u32
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

memcpy_dtype = MemcpyDataType.MEMCPY_16BIT
runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

sym_weight = runner.get_id("weight")
sym_sqrt_diag_A = runner.get_id("sqrt_diag_A")

runner.load()
runner.run()

A = np.array([[42.0, 42.0, 42.0, 42.0, 42.0],
              [42.0, 42.0, 42.0, 42.0, 42.0],
              [42.0, 42.0, 42.0, 42.0, 42.0],
              [42.0, 42.0, 42.0, 42.0, 42.0],
              [42.0, 42.0, 42.0, 42.0, 42.0]]).astype(np.float16)
B = np.array([10, 20, 30, 40, 50]).astype(np.int16)

def transformation(value: np.array, coeff1: float, coeff2: float, weight: np.array):
  return np.multiply(value, coeff1 + weight) + np.multiply(value, coeff2 + weight)

def reduction(array):
  return sum(array)

np.random.seed(seed=7)

print("step 1: copy mode H2D")
weights = np.random.random(size).astype(np.float16)
tensors_u32 = input_array_to_u32(weights, 0, size)
runner.memcpy_h2d(sym_weight, tensors_u32, 0, 0, 1, 1, size, \
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=True)

print("stpe 2: call f_run to test @map")
runner.launch("f_run", nonblock=False)

print("step 3: copy mode D2H")
# The D2H buffer must be of type u32
out_tensors_u32 = np.zeros(size, np.uint32)
runner.memcpy_d2h(out_tensors_u32, sym_sqrt_diag_A, 0, 0, 1, 1, size, \
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)
# remove upper 16-bit of each u32
sqrt_result = memcpy_view(out_tensors_u32, np.dtype(np.float16))

runner.stop()

expected = np.sqrt(np.diag(A))
np.testing.assert_equal(sqrt_result, expected)

debug_mod = debug_util(dirname, cmaddr=args.cmaddr)
core_offset_x = 4
core_offset_y = 1
print(f"=== dump core: core rectangle starts at {core_offset_x}, {core_offset_y}")

# Transformation example
expected = transformation(np.diag(A), 2.0, 6.0, weights)
np.fill_diagonal(A, expected)
actual = debug_mod.get_symbol(core_offset_x, core_offset_y, "A", np.float16)
np.testing.assert_equal(actual.reshape((5, 5)), A)

# Reduction example
sum_result = np.array([reduction(B)], dtype=np.int16)
expected = debug_mod.get_symbol(core_offset_x, core_offset_y, "sum", np.int16)
np.testing.assert_equal(sum_result, expected)

print("SUCCESS!")
