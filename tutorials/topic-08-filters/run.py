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
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder # pylint: disable=no-name-in-module

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='the test name')
parser.add_argument("--cmaddr", help="IP:port for CS system")
args = parser.parse_args()
dirname = args.name

runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

result_symbol = runner.get_id('result')

runner.load()
runner.run()

num_recv_pes = 3 # 3 PEs receive from the sender
elems_per_pe = 3 # Each recv PE receives 3 elems after filtering

print("step 1: launch function to send data to neighbors")
runner.launch("main_fn", nonblock=False)

print("step 2: copy back data from receiving PEs")
result = np.zeros(num_recv_pes * elems_per_pe, np.float32)
runner.memcpy_d2h(result, result_symbol, 1, 0, num_recv_pes, 1, elems_per_pe, streaming=False, \
   data_type=MemcpyDataType.MEMCPY_32BIT, order=MemcpyOrder.ROW_MAJOR, nonblock=False)

runner.stop()

oracle = [6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5]
np.testing.assert_allclose(result, oracle, atol=0.0001, rtol=0)
print("SUCCESS!")
