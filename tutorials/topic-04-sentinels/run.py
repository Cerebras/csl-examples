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

from cerebras.sdk.sdk_utils import memcpy_view, input_array_to_u32
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder # pylint: disable=no-name-in-module

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='the test name')
parser.add_argument("--cmaddr", help="IP:port for CS system")
args = parser.parse_args()
dirname = args.name

# Parse the compile metadata
with open(f"{dirname}/out.json", encoding="utf-8") as json_file:
  compile_data = json.load(json_file)
params = compile_data["params"]
MEMCPYH2D_DATA_1 = int(params["MEMCPYH2D_DATA_1_ID"])
MEMCPYH2D_DATA_2 = int(params["MEMCPYH2D_DATA_2_ID"])
MEMCPYD2H_DATA_1 = int(params["MEMCPYD2H_DATA_1_ID"])
size = int(params["size"])
print(f"MEMCPYH2D_DATA_1 = {MEMCPYH2D_DATA_1}")
print(f"MEMCPYH2D_DATA_2 = {MEMCPYH2D_DATA_2}")
print(f"MEMCPYD2H_DATA_1 = {MEMCPYD2H_DATA_1}")
print(f"size = number of PEs in a column = {size}")

# memcpy_dtype is DON'T care under streaming mode
memcpy_dtype = MemcpyDataType.MEMCPY_16BIT
runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

runner.load()
runner.run()

num_wvlts = 11
print(f"num_wvlts = number of wavelets for each PE = {num_wvlts}")

print("step 1: streaming H2D_1 sends number of input wavelets to P0.0")
h2d1_u32 = np.ones(size).astype(np.uint32) * num_wvlts
runner.memcpy_h2d(MEMCPYH2D_DATA_1, h2d1_u32.ravel(), 0, 0, 1, size, 1, \
    streaming=True, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=True)

# Use a deterministic seed so that CI results are predictable
np.random.seed(seed=7)

# Setup a {size}x11 input tensor that is reduced along the second dimension
input_tensor = np.random.rand(size, num_wvlts).astype(np.float16)
expected = np.sum(input_tensor, axis=1)

print("step 2: streaming H2D_2 to P0.0")
# "input_tensor" is a 1d array
# The type of input_tensor is float16, we need to extend it to uint32
# There are two kind of extension when using the utility function input_array_to_u32
#    input_array_to_u32(np_arr: np.ndarray, sentinel: Optional[int], fast_dim_sz: int)
# 1) zero extension:
#    sentinel = None
# 2) upper 16-bit is the index of the array:
#    sentinel is Not None
#
# In this example, the upper 16-bit is don't care because pe_program.csl only
# reads lower 16-bit
tensors_u32 = input_array_to_u32(input_tensor.ravel(), 1, num_wvlts)
runner.memcpy_h2d(MEMCPYH2D_DATA_2, tensors_u32, 0, 0, 1, size, num_wvlts, \
    streaming=True, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=True)

print("step 3: streaming D2H at P3.0")
# The D2H buffer must be of type u32
out_tensors_u32 = np.zeros(size, np.uint32)
runner.memcpy_d2h(out_tensors_u32, MEMCPYD2H_DATA_1, 3, 0, 1, size, 1, \
    streaming=True, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)
# remove upper 16-bit of each u32
result_tensor = memcpy_view(out_tensors_u32, np.dtype(np.float16))

runner.stop()

# Ensure that the result matches our expectation
np.testing.assert_allclose(result_tensor, expected, atol=0.05, rtol=0)
print("SUCCESS!")
