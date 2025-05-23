#!/usr/bin/env cs_python

# Copyright 2025 Cerebras Systems.
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

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

sum_symbol = runner.get_id("sum")

runner.load()
runner.run()

num_elems = 5
x = np.arange(num_elems, dtype=np.uint32)

print("step 1: streaming H2D to 1st PE")
runner.memcpy_h2d(MEMCPYH2D_DATA_1, x, 0, 0, 1, 1, num_elems, \
    streaming=True, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=True)

print("step 2: copy mode D2H buf")
result = np.zeros(width, np.uint32)
runner.memcpy_d2h(result, sum_symbol, 0, 0, width, 1, 1, \
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)

runner.stop()

# In order, the host streams in 0, 1, 2, 3, 4 from the West.
# Red tasks add values to running global sum on its PE.
# Blue tasks add 2*values to running global sum on its PE.
# PEs 0, 2 activate blue task; 1, 3 activate red task.
# Final value of sum var on even PEs will be: 20
# Final value of sum var on odd PEs will be: 10
oracle = np.zeros([width], dtype=np.uint32)
for i in range(width):
  oracle[i] = ((i+1) % 2 + 1) * num_elems * (num_elems-1) / 2

# Assert that all values of 'sum' are as expected
np.testing.assert_equal(result, oracle)
print("SUCCESS!")
