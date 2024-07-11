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

runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

result_symbol = runner.get_id('result')
start_ts_symbol = runner.get_id('start_timestamp')
finish_ts_symbol = runner.get_id('finish_timestamp')

runner.load()
runner.run()

print("step 1: call f_run to start computation")
runner.launch("f_run", nonblock=False)

print("step 2: copy back result")
# The D2H buffer must be of type u32
result = np.zeros(1, np.float32)
runner.memcpy_d2h(result, result_symbol, 0, 0, 1, 1, 1, \
    streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT, \
    order=MemcpyOrder.COL_MAJOR, nonblock=False)

print("step 3: copy back timestamps")
# The D2H buffer must be of type u32
start_timestamps_u32 = np.zeros(3, np.uint32)
runner.memcpy_d2h(start_timestamps_u32, start_ts_symbol, 0, 0, 1, 1, 3, \
    streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT, \
    order=MemcpyOrder.COL_MAJOR, nonblock=False)

finish_timestamps_u32 = np.zeros(3, np.uint32)
runner.memcpy_d2h(finish_timestamps_u32, finish_ts_symbol, 0, 0, 1, 1, 3, \
    streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT, \
    order=MemcpyOrder.COL_MAJOR, nonblock=False)

# remove upper 16-bit of each u32
start_timestamps = memcpy_view(start_timestamps_u32, np.dtype(np.uint16))
finish_timestamps = memcpy_view(finish_timestamps_u32, np.dtype(np.uint16))

runner.stop()

# Helper functions for computing the delta in the cycle count
def make_u48(words):
  return words[0] + (words[1] << 16) + (words[2] << 32)

def subtract_timestamps(finish, start):
  return make_u48(finish) - make_u48(start)

cycles = subtract_timestamps(finish_timestamps, start_timestamps)
print("cycle count:", cycles)

print(f"result = {result}, np.pi = {np.pi}, tol = {args.tolerance}")
np.testing.assert_allclose(result, np.pi, atol=args.tolerance, rtol=0)
print("SUCCESS!")
