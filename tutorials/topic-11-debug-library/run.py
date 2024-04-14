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

from cerebras.sdk.debug.debug_util import debug_util
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
width = int(params["width"])
print(f"MEMCPYH2D_DATA_1 = {MEMCPYH2D_DATA_1}")
print(f"width = {width}")

memcpy_dtype = MemcpyDataType.MEMCPY_16BIT
runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

sym_buf = runner.get_id("buf")

runner.load()
runner.run()

num_entries = 4
x = np.arange(num_entries, dtype=np.int16)

print("step 1: streaming H2D to 1st PE")
tensors_u32 = input_array_to_u32(x, 0, num_entries)
runner.memcpy_h2d(MEMCPYH2D_DATA_1, tensors_u32, 0, 0, 1, 1, num_entries, \
    streaming=True, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=True)

print("step 2: copy mode D2H buf (need at least one D2H)")
# The D2H buffer must be of type u32
out_tensors_u32 = np.zeros(1, np.uint32)
runner.memcpy_d2h(out_tensors_u32, sym_buf, 0, 0, 1, 1, 1, \
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)
# remove upper 16-bit of each u32
buf_result = memcpy_view(out_tensors_u32, np.dtype(np.int16))

runner.stop()

debug_mod = debug_util(dirname, cmaddr=args.cmaddr)
core_offset_x = 4
core_offset_y = 1
print(f"=== dump core: core rectangle starts at {core_offset_x}, {core_offset_y}")

result = np.zeros([width, num_entries])
for idx in range(width):
  # Get traces recorded in 'trace'
  trace_output = debug_mod.read_trace(core_offset_x + idx, core_offset_y, 'trace')

  # Copy all recorded trace values of variable 'global'
  result[idx, :] = trace_output[1::2]

  # Get timestamp traces recorded in 'times'
  timestamp_output = debug_mod.read_trace(core_offset_x + idx, core_offset_y, 'times')

  # Print out all traces for PE
  print("PE (", idx, ", 0): ")
  print("Trace: ", trace_output)
  print("Times: ", timestamp_output)
  print()

# In order, the host streams in 0, 1, 2, 3 from the West.
# Red tasks add values to running global sum on its PE.
# Blue tasks add 2*values to running global sum on its PE.
# Value of global var is recorded after each update.
# PEs 0, 2 activate blue task; 1, 3 activate red task.
# Trace values of global var on even PEs will be: 0, 2, 6, 12
# Trace values of global var on odd PEs will be: 0, 1, 3, 6
oracle = np.empty([width, num_entries])
for i in range(width):
  for j in range(num_entries):
    oracle[i, j] = ((i+1) % 2 + 1) * j * (j+1) / 2

# Assert that all trace values of 'global' are as expected
np.testing.assert_equal(result, oracle)
print("SUCCESS!")
