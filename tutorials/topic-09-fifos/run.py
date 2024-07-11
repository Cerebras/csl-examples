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
parser.add_argument("--cmaddr", help="IP:port for CS system")
args = parser.parse_args()
dirname = args.name

# Parse the compile metadata
with open(f"{dirname}/out.json", encoding="utf-8") as json_file:
  compile_data = json.load(json_file)
params = compile_data["params"]
MEMCPYH2D_DATA_1 = int(params["MEMCPYH2D_DATA_1_ID"])
MEMCPYD2H_DATA_1 = int(params["MEMCPYD2H_DATA_1_ID"])
size = int(params["num_elems_to_process"])
print(f"MEMCPYH2D_DATA_1 = {MEMCPYH2D_DATA_1}")
print(f"MEMCPYD2H_DATA_1 = {MEMCPYD2H_DATA_1}")
print(f"size = {size}")

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

runner.load()
runner.run()

np.random.seed(seed=7)

input_tensor = np.random.random(size).astype(np.float32)
print("step 1: streaming H2D")
runner.memcpy_h2d(MEMCPYH2D_DATA_1, input_tensor, 0, 0, 1, 1, size, \
    streaming=True, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=True)

print("step 2: streaming D2H")
result_tensor = np.zeros(size, np.float32)
runner.memcpy_d2h(result_tensor, MEMCPYD2H_DATA_1, 0, 0, 1, 1, size, \
    streaming=True, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)

runner.stop()

add_ten_negate = -(input_tensor + 10.0)
expected = add_ten_negate * add_ten_negate * add_ten_negate

np.testing.assert_equal(result_tensor, expected)
print("SUCCESS!")
