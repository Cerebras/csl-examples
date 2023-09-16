#!/usr/bin/env cs_python

# Copyright 2022 Cerebras Systems.
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

from cerebras.sdk.sdk_utils import memcpy_view
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
print(f"MEMCPYH2D_DATA_1 = {MEMCPYH2D_DATA_1}")
print(f"MEMCPYD2H_DATA_1 = {MEMCPYD2H_DATA_1}")

memcpy_dtype = MemcpyDataType.MEMCPY_16BIT
runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

runner.load()
runner.run()

# Turn each tuple of two 16-bit integers into one 32-bit integer
packed = [(idx << 16) + val for idx, val in [(0, 42), (3, 26)]]
packed_tensor = np.array(packed, dtype=np.int32)

print("step 1: streaming H2D")
# "packed_tensor" must be an 1d array of type u32
runner.memcpy_h2d(MEMCPYH2D_DATA_1, packed_tensor, 0, 0, 1, 1, 2, \
    streaming=True, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=True)

print("step 2: streaming D2H")
# The D2H buffer must be of type u32
out_tensors_u32 = np.zeros(2, np.uint32)
runner.memcpy_d2h(out_tensors_u32, MEMCPYD2H_DATA_1, 0, 0, 1, 1, 2, \
    streaming=True, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)
# remove upper 16-bit of each u32
result_tensor = memcpy_view(out_tensors_u32, np.dtype(np.int16))

runner.stop()

# Ensure that the result matches our expectation
# Since zero wavelets are skipped during transmission, the `@mov16` operation
# in the code is executed only twice, once for each non-zero wavelet data
np.testing.assert_equal(result_tensor, [42, 26])
print("SUCCESS!")
