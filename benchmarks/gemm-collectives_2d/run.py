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

from cerebras.sdk.runtime import runtime_utils
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime     # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder    # pylint: disable=no-name-in-module

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="the test name")
parser.add_argument("--cmaddr", help="IP:port for CS system")
args = parser.parse_args()

# Get params from compile metadata
with open(f"{args.name}/out.json", encoding='utf-8') as json_file:
  compile_data = json.load(json_file)

# Kernel rectangle and per-PE matrix dimensions
P  = int(compile_data['params']['P'])
Mt = int(compile_data['params']['Mt'])
Kt = int(compile_data['params']['Kt'])
Nt = int(compile_data['params']['Nt'])

# Full matrix dimensions
# A is M x K, B is K x N, C is M x N
M = Mt * P
K = Kt * P
N = Nt * P

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
memcpy_order = MemcpyOrder.ROW_MAJOR

A = np.arange(M * K, dtype=np.float32).reshape((M, K))
B = np.arange(K * N, dtype=np.float32).reshape((K, N))

simulator = SdkRuntime(args.name, cmaddr=args.cmaddr)

symbol_A = simulator.get_id("A")
symbol_B = simulator.get_id("B")
symbol_C = simulator.get_id("C")

simulator.load()
simulator.run()


iportmap_A = f"{{ A[j=0:{M-1}][i=0:{K-1}] -> [PE[i//{Kt}, j//{Mt}] -> \
    index[i%{Kt}, j%{Mt}]] }}"

(px, py, w, h, l, data) = runtime_utils.convert_input_tensor(iportmap_A, A)
simulator.memcpy_h2d(symbol_A, data, px, py, w, h, l,
                     streaming=False, data_type=memcpy_dtype, nonblock=False,
                     order=memcpy_order)


iportmap_B = f"{{ B[j=0:{K-1}][i=0:{N-1}] -> [PE[i//{Nt}, j//{Kt}] -> \
    index[i%{Nt}, j%{Kt}]] }}"

(px, py, w, h, l, data) = runtime_utils.convert_input_tensor(iportmap_B, B)
simulator.memcpy_h2d(symbol_B, data, px, py, w, h, l,
                     streaming=False, data_type=memcpy_dtype, nonblock=False,
                     order=memcpy_order)


simulator.call("main", [], nonblock=False)


oportmap_C = f"{{ C[j=0:{M-1}][i=0:{N-1}] -> [PE[i//{Nt}, j//{Mt}] -> \
    index[i%{Nt}, j%{Mt}]] }}"

(px, py, w, h, l, data) = runtime_utils.prepare_output_tensor(oportmap_C, np.float32)
simulator.memcpy_d2h(data, symbol_C, px, py, w, h, l,
                     streaming=False, data_type=memcpy_dtype, nonblock=False,
                     order=memcpy_order)
C = runtime_utils.format_output_tensor(oportmap_C, np.float32, data)


simulator.stop()

# Check the result
C_expected = np.dot(A, B)

np.testing.assert_equal(C_expected, C)

print("SUCCESS")
