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

from cerebras.sdk.sdk_utils import memcpy_view, input_array_to_u32
from cerebras.sdk.debug.debug_util import debug_util
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder # pylint: disable=no-name-in-module

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="the test name")
parser.add_argument("--cmaddr", help="IP:port for CS system")
parser.add_argument("--debug", help="debug", action="store_true")
args = parser.parse_args()
dirname = args.name

# Parse the compile metadata
with open(f"{dirname}/out.json", encoding="utf-8") as json_file:
  compile_data = json.load(json_file)
compile_params = compile_data["params"]
kernel_rows = int(compile_params["kernel_rows"]) # Height of kernel
kernel_cols = int(compile_params["kernel_cols"]) # Width of kernel
A_rows = int(compile_params["matrix_rows"]) # number of rows of A
A_cols = int(compile_params["matrix_cols"]) # number of columns of A

MEMCPYH2D_DATA_1 = int(compile_params["MEMCPYH2D_DATA_1_ID"])
MEMCPYH2D_DATA_2 = int(compile_params["MEMCPYH2D_DATA_2_ID"])
MEMCPYD2H_DATA_1 = int(compile_params["MEMCPYD2H_DATA_1_ID"])
print(f"MEMCPYH2D_DATA_1 = {MEMCPYH2D_DATA_1}")
print(f"MEMCPYH2D_DATA_2 = {MEMCPYH2D_DATA_2}")
print(f"MEMCPYD2H_DATA_1 = {MEMCPYD2H_DATA_1}")

print(f"Program runs on a {kernel_cols}x{kernel_rows} rectangle of PEs")

# Create tensors for A, X, B.
print(f"the matrix A is A_rows-by-A_cols, A_cols = {A_cols}, A_rows = {A_rows}")

X_rows = A_cols
X_cols = 1

B_rows = A_rows
B_cols = 1

# Use a deterministic seed so that CI results are predictable
np.random.seed(seed=7)

if args.debug:
  A = np.arange(A_rows*A_cols).reshape(A_rows, A_cols).astype(np.float16)
  X = np.arange(X_rows*X_cols).reshape(X_rows, X_cols).astype(np.float16)
  B = np.zeros((B_rows, B_cols), np.float16)
else:
  A = np.random.rand(A_rows, A_cols).astype(np.float16)
  X = np.random.rand(X_rows, X_cols).astype(np.float16)
  B = np.random.rand(B_rows, B_cols).astype(np.float16)

if args.debug:
  print(f"A = {A}")
  print(f"X = {X}")
  print(f"B = {B}")

# Compute expected result
expected = (A @ X) + B

memcpy_dtype = MemcpyDataType.MEMCPY_16BIT
runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

sym_A = runner.get_id("A")

runner.load()
runner.run()

# Split A tensor across PEs
# A[M, N] -> kernel_cols * kernel_rows * A[M // kernel_cols, N // kernel_rows]
per_pe_rows = A_rows // kernel_rows
per_pe_cols = A_cols // kernel_cols

print(f"the local size of y: per_pe_rows = {per_pe_rows}")
print(f"the local size of x: per_pe_cols = {per_pe_cols}")

# Example: w=2, h=2, A is 4-by-4
# A = |  0  1  2  3 |
#     |  4  5  6  7 |
#     |  8  9 10 11 |
#     | 12 13 14 15 |
# A1 = A.reshape(2,2,2,2)
# A1 = | | 0  1|  | 4  5| |
#      | | 2  3|, | 6  7| |
#      |                  |
#      | | 8  9|  |12 13| |
#      | |10 11|, |14 15| |
# A2 = A1.transpose(0, 2, 1, 3)
# A2 = | | 0  1|  | 2  3| |
#      | | 4  5|, | 6  7| |
#      |                  |
#      | | 8  9|  |10 11| |
#      | |12 13|, |14 15| |
# A3 = A2.reshape(2,2,4)
# A3 = |  0  1  4  5 |
#      |  2  3  6  7 |
#      |  8  9 12 13 |
#      | 10 11 14 15 |
# A3 is h-w-l
A1 = A.reshape(kernel_rows, per_pe_rows,
               kernel_cols, per_pe_cols)
A2 = A1.transpose(0, 2, 1, 3)
A3 = A2.reshape(kernel_rows, kernel_cols, per_pe_rows*per_pe_cols)
print("step 1: copy mode H2D A")
A_1d_u32 = input_array_to_u32(np_arr=A3.ravel(), sentinel=0, \
    fast_dim_sz=per_pe_rows*per_pe_cols)
runner.memcpy_h2d(sym_A, A_1d_u32, 0, 0, kernel_cols, kernel_rows, per_pe_rows*per_pe_cols, \
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=True)

print("step 2: streaming mode H2D X at 1st row via color MEMCPYH2D_DATA_1")
print("    each PE receives x, performs local A*x and triggers chain reduction")
# extend x with index in the upper 16-bit
x_1d_u32 = input_array_to_u32(np_arr=X.ravel(), sentinel=1, fast_dim_sz=per_pe_cols)
runner.memcpy_h2d(MEMCPYH2D_DATA_1, x_1d_u32, 0, 0, kernel_cols, 1, per_pe_cols,\
    streaming=True, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=True)

print("step 3: streaming mode H2D B at 1st column via color MEMCPYH2D_DATA_2")
print("   1st column receives B to start the chain reduction, others wait for data from the west")
# extend x with zero in the upper 16-bit
b_1d_u32 = input_array_to_u32(np_arr=B.ravel(), sentinel=0, fast_dim_sz=per_pe_rows)
runner.memcpy_h2d(MEMCPYH2D_DATA_2, b_1d_u32, 0, 0, 1, kernel_rows, per_pe_rows,\
    streaming=True, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=True)

print("step 4: streaming mode D2H y at last column via color MEMCPYD2H_DATA_1")
print("   this D2H indidates the y = A*x is done")
y_1d_u32 = np.zeros(B_rows, np.uint32)
runner.memcpy_d2h(y_1d_u32, MEMCPYD2H_DATA_1, kernel_cols-1, 0, 1, kernel_rows, per_pe_rows, \
    streaming=True, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
# remove upper 16-bit of each u32
result_tensor = memcpy_view(y_1d_u32, np.dtype(np.float16))
result_tensor = result_tensor.reshape(B_rows, B_cols)

runner.stop()

np.testing.assert_allclose(result_tensor, expected, atol=0.01, rtol=0)
print("SUCCESS!")

if args.debug:
  debug_mod = debug_util(dirname, cmaddr=args.cmaddr)
  core_fabric_offset_x = 4
  core_fabric_offset_y = 1
  print(f"=== dump core: core_fabric {core_fabric_offset_x}, {core_fabric_offset_y}")

  # local A*x
  Ax_hwl = np.zeros((kernel_rows, kernel_cols, per_pe_rows), np.float16)
  for py in range(kernel_rows):
    for px in range(kernel_cols):
      t = debug_mod.get_symbol(core_fabric_offset_x+px, core_fabric_offset_y+py,\
         'mul_temp', np.float16)
      Ax_hwl[py, px, :] = t
  print(f"Ax_hwl = \n{Ax_hwl}")

  x_hwl = np.zeros((kernel_rows, kernel_cols, per_pe_cols), np.float16)
  for py in range(kernel_rows):
    for px in range(kernel_cols):
      t = debug_mod.get_symbol(core_fabric_offset_x+px, core_fabric_offset_y+py,\
         'x_temp', np.float16)
      x_hwl[py, px, :] = t
  print(f"x_hwl = \n{x_hwl}")

  num_recv_x_hwl = np.zeros((kernel_rows, kernel_cols, 1), np.int16)
  for py in range(kernel_rows):
    for px in range(kernel_cols):
      t = debug_mod.get_symbol(core_fabric_offset_x+px, core_fabric_offset_y+py,\
         'num_recv_x', np.int16)
      num_recv_x_hwl[py, px, :] = t
  print(f"num_recv_x_hwl = \n{num_recv_x_hwl}")
