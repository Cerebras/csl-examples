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
num_elems = int(params["num_elems"])
width = int(params["width"])
print(f"width = {width}")
print(f"num_elems = {num_elems}")

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

sym_buf = runner.get_id("buf")

runner.load()
runner.run()

x = np.arange(num_elems, dtype=np.uint32)

print("step 1: H2D copy buf to sender PE")
runner.memcpy_h2d(sym_buf, x, 0, 0, 1, 1, num_elems, \
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=False)

print("step 2: launch main_fn")
runner.launch('main_fn', nonblock=False)

print("step 3: D2H copy buf back from all PEs")
out_buf = np.arange(width*num_elems, dtype=np.uint32)
runner.memcpy_d2h(out_buf, sym_buf, 0, 0, width, 1, num_elems, \
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=False)

runner.stop()

# Receiver PEs write received value to out_buf,
# so out_buf of each PE should be same as x
# Assert that out_buf of each PE matches input array x
np.testing.assert_equal(np.tile(x, (width,1)), out_buf.reshape(width, num_elems))
print("SUCCESS!")
