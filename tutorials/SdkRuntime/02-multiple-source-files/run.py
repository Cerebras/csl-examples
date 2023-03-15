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
import numpy as np

from cerebras.sdk.runtime import runtime_utils
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='the test name')
parser.add_argument("--cmaddr", help="IP:port for CS system")
args = parser.parse_args()
name = args.name

# Construct a runner using SdkRuntime
runner = SdkRuntime(name, cmaddr=args.cmaddr)

# Color used for memcpy D2H streaming
output_color = 1

# Output will be a 16-bit data type
memcpy_dtype = MemcpyDataType.MEMCPY_16BIT
memcpy_order = MemcpyOrder.ROW_MAJOR

# ISL map to indicate PE that will stream back output with D2H memcpy
output_port_map = "{out_tensor[idx=0:0] -> [PE[0,0] -> index[idx]]}"

# Load and run the program
runner.load()
runner.run()

# Prepare output tensor, perform streaming D2H memcpy, and format result
(px, py, w, h, l, data) \
  = runtime_utils.prepare_output_tensor(output_port_map, np.int16)

runner.memcpy_d2h(data, output_color, px, py, w, h, l,
                  streaming=True, data_type=memcpy_dtype,
                  order=memcpy_order, nonblock=False)

result = runtime_utils.format_output_tensor(output_port_map, np.int16, data)

# Stop the program
runner.stop()

# Ensure that the result matches our expectation
np.testing.assert_equal(result, [42])
print("SUCCESS!")
