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
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="the test name")
parser.add_argument("--cmaddr", help="IP:port for CS system")
args = parser.parse_args()

# Initialize the input matrices
P, Nt, Kt, Mt = 4, 14, 14, 14

N = Nt * P
K = Kt * P
M = Mt * P

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
memcpy_order = MemcpyOrder.ROW_MAJOR

A = np.arange(N * K, dtype=np.float32).reshape((N, K)) % 100
B = np.arange(K * M, dtype=np.float32).reshape((K, M)) % 100

simulator = SdkRuntime(args.name, cmaddr=args.cmaddr)

symbol_A = simulator.get_id("A")
symbol_B = simulator.get_id("B")
symbol_C = simulator.get_id("C")

simulator.load()
simulator.run()


iportmap_A = f"{{ A[j=0:{N-1}][i=0:{K-1}] -> [PE[i//{Kt}, j//{Nt}] -> \
    index[i%{Kt}, j%{Nt}]] }}"

(px, py, w, h, l, data) = runtime_utils.convert_input_tensor(iportmap_A, A)
simulator.memcpy_h2d(symbol_A, data, px, py, w, h, l,
                     streaming=False, data_type=memcpy_dtype, nonblock=False,
                     order=memcpy_order)


iportmap_B = f"{{ B[j=0:{K-1}][i=0:{M-1}] -> [PE[i//{Mt}, j//{Kt}] -> \
    index[i%{Mt}, j%{Kt}]] }}"

(px, py, w, h, l, data) = runtime_utils.convert_input_tensor(iportmap_B, B)
simulator.memcpy_h2d(symbol_B, data, px, py, w, h, l,
                     streaming=False, data_type=memcpy_dtype, nonblock=False,
                     order=memcpy_order)


simulator.call("main", [], nonblock=False)


oportmap_C = f"{{ C[j=0:{N-1}][i=0:{M-1}] -> [PE[i//{Mt}, j//{Nt}] -> \
    index[i%{Mt}, j%{Nt}]] }}"

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
