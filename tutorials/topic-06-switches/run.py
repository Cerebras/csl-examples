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

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder # pylint: disable=no-name-in-module

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='the test name')
parser.add_argument("--cmaddr", help="IP:port for CS system")
args = parser.parse_args()
dirname = args.name

runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

# Get symbol for copying recv results off device
result_symbol = runner.get_id('result')

runner.load()
runner.run()

runner.launch('main_fn', nonblock=False)

# Copy arr back from PEs that received wlts
west_result = np.zeros([1], dtype=np.uint32)
runner.memcpy_d2h(west_result, result_symbol, 0, 1, 1, 1, 1, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

east_result = np.zeros([1], dtype=np.uint32)
runner.memcpy_d2h(east_result, result_symbol, 2, 1, 1, 1, 1, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

south_result = np.zeros([1], dtype=np.uint32)
runner.memcpy_d2h(south_result, result_symbol, 1, 2, 1, 1, 1, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

north_result = np.zeros([1], dtype=np.uint32)
runner.memcpy_d2h(north_result, result_symbol, 1, 0, 1, 1, 1, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

runner.stop()

print("East result: ", east_result)
print("South result: ", south_result)
print("North result: ", north_result)
print("West result: ", west_result)

np.testing.assert_equal(0, east_result)
np.testing.assert_equal(2, south_result)
np.testing.assert_equal(4, north_result)
np.testing.assert_equal(6, west_result)

print("SUCCESS!")
