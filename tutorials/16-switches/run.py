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

from cerebras.sdk.debug.debug_util import debug_util
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
MEMCPYD2H_DATA_1 = int(params["MEMCPYD2H_DATA_1_ID"])
print(f"MEMCPYD2H_DATA_1 = {MEMCPYD2H_DATA_1}")

memcpy_dtype = MemcpyDataType.MEMCPY_16BIT
runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

runner.load()
runner.run()

print("step 1: call f_run to start at PE1.1")
runner.launch("f_run", nonblock=False)

print("step 2: streaming D2H at P1.0 (end of communication)")
# The D2H buffer must be of type u32
out_tensors_u32 = np.zeros(1, np.uint32)
runner.memcpy_d2h(out_tensors_u32, MEMCPYD2H_DATA_1, 1, 0, 1, 1, 1, \
    streaming=True, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)
# remove upper 16-bit of each u32
result_tensor = memcpy_view(out_tensors_u32, np.dtype(np.int16))

runner.stop()

debug_mod = debug_util(dirname, cmaddr=args.cmaddr)
core_offset_x = 4
core_offset_y = 1
print(f"=== core rectangle starts at {core_offset_x}, {core_offset_y}")
# sender PE is P1.1
# top PE of sender PE is P1.0
result_top = debug_mod.get_symbol(core_offset_x+1, core_offset_y+0, "global", np.uint16)
# left PE of sender PE is P0.1
result_left = debug_mod.get_symbol(core_offset_x+0, core_offset_y+1, "global", np.uint16)
# right PE of sender PE is P2.1
result_right = debug_mod.get_symbol(core_offset_x+2, core_offset_y+1, "global", np.uint16)
# bottom PE of sender PE is P1.2
result_bottom = debug_mod.get_symbol(core_offset_x+1, core_offset_y+2, "global", np.uint16)

np.testing.assert_allclose(result_top, 0xdd)
np.testing.assert_allclose(result_left, 0xaa)
np.testing.assert_allclose(result_right, 0xbb)
np.testing.assert_allclose(result_bottom, 0xcc)
print("SUCCESS!")
