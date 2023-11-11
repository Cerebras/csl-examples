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

from cerebras.sdk.sdk_utils import memcpy_view
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder # pylint: disable=no-name-in-module

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='the test name')
parser.add_argument("--cmaddr", help="IP:port for CS system")
parser.add_argument("--tolerance", type=float, help="tolerance for result")
args = parser.parse_args()
dirname = args.name

# Parse the compile metadata
with open(f"{dirname}/out.json", encoding="utf-8") as json_file:
  compile_data = json.load(json_file)
params = compile_data["params"]
MEMCPYD2H_DATA_1 = int(params["MEMCPYD2H_DATA_1_ID"])
print(f"MEMCPYD2H_DATA_1 = {MEMCPYD2H_DATA_1}")

print("The simfab may take 25 sec more")
runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

runner.load()
runner.run()

print("step 1: call f_run to start streaming D2H (result)")
runner.launch("f_run", nonblock=False)

print("step 2: streaming D2H (result)")
# The D2H buffer must be of type u32
result = np.zeros(1, np.float32)
runner.memcpy_d2h(result, MEMCPYD2H_DATA_1, 0, 0, 1, 1, 1, \
    streaming=True, data_type=MemcpyDataType.MEMCPY_32BIT, \
    order=MemcpyOrder.COL_MAJOR, nonblock=False)

print("step 3: call f_send_timestamps to start streaming D2H (timestamp)")
runner.launch("f_send_timestamps", nonblock=False)

print("step 4: streaming D2H (timestamps)")
# The D2H buffer must be of type u32
timestamps_u32 = np.zeros(6, np.uint32)
runner.memcpy_d2h(timestamps_u32, MEMCPYD2H_DATA_1, 0, 0, 1, 1, 6, \
    streaming=True, data_type=MemcpyDataType.MEMCPY_16BIT, \
    order=MemcpyOrder.COL_MAJOR, nonblock=False)
# remove upper 16-bit of each u32
timestamps = memcpy_view(timestamps_u32, np.dtype(np.uint16))

runner.stop()

# Helper functions for computing the delta in the cycle count
def make_u48(words):
  return words[0] + (words[1] << 16) + (words[2] << 32)

def subtract_timestamps(words):
  return make_u48(words[3:]) - make_u48(words[0:3])

cycles = subtract_timestamps(timestamps)
print("cycle count:", cycles)

print(f"result = {result}, np.pi = {np.pi}, tol = {args.tolerance}")
np.testing.assert_allclose(result, np.pi, atol=args.tolerance, rtol=0)
print("SUCCESS!")
