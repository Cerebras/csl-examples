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

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder # pylint: disable=no-name-in-module

# Read arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', help="the test compile output dir")
parser.add_argument('--cmaddr', help="IP:port for CS system")
args = parser.parse_args()

# Get matrix dimensions from compile metadata
with open(f"{args.name}/out.json", encoding='utf-8') as json_file:
  compile_data = json.load(json_file)

# Matrix dimensions
N = int(compile_data['params']['N']) # columns
M = int(compile_data['params']['M']) # rows

# PE grid dimensions
kernel_x_dim = int(compile_data['params']['kernel_x_dim'])
kernel_y_dim = int(compile_data['params']['kernel_y_dim'])

# Colors used for memcpy streaming
MEMCPYH2D_DATA_1 = int(compile_data['params']['MEMCPYH2D_DATA_1_ID'])
MEMCPYH2D_DATA_2 = int(compile_data['params']['MEMCPYH2D_DATA_2_ID'])
MEMCPYD2H_DATA_1 = int(compile_data['params']['MEMCPYD2H_DATA_1_ID'])

# Construct A, x, b
A = np.arange(M*N, dtype=np.float32).reshape(M,N)
x = np.full(shape=N, fill_value=1.0, dtype=np.float32)
b = np.full(shape=M, fill_value=2.0, dtype=np.float32)

# Calculate expected y
y_expected = A@x + b

# Size of N dimension on each PE
N_per_PE = N // kernel_x_dim
M_per_PE = M // kernel_y_dim

# Construct a runner using SdkRuntime
runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

# Get symbol for A on device
A_symbol = runner.get_id('A')

# Load and run the program
runner.load()
runner.run()

# Copy chunks of A into all PEs
# Each chunk on each PE is stored column major
A_prepared = A.reshape(kernel_y_dim, M_per_PE, kernel_x_dim, N_per_PE).transpose(0, 2, 3, 1).ravel()
runner.memcpy_h2d(A_symbol, A_prepared, 0, 0, kernel_x_dim, kernel_y_dim, M_per_PE*N_per_PE,
  streaming=False, order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT,
  nonblock=False)

# Stream x into PEs (0, 0) and (kernel_x_dim-1, 0)
# PE (0, 0) gets first N/2 elements; PE (1, 0) gets last N/2 elements
runner.memcpy_h2d(MEMCPYH2D_DATA_1, x, 0, 0, kernel_x_dim, 1, N_per_PE, streaming=True,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# Stream b into PEs (0, 0) to (0, kernel_y_dim-1)
runner.memcpy_h2d(MEMCPYH2D_DATA_2, b, 0, 0, 1, kernel_y_dim, M_per_PE, streaming=True,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# Stream y back from PEs (kernel_x_dim-1, 0) and (kernel_x_dim-1, kernel_y_dim-1)
y_result = np.zeros([M], dtype=np.float32)
runner.memcpy_d2h(y_result, MEMCPYD2H_DATA_1, kernel_x_dim-1, 0, 1, kernel_y_dim, M_per_PE,
  streaming=True, order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT,
  nonblock=False)

# Stop the program
runner.stop()

# Ensure that the result matches our expectation
np.testing.assert_allclose(y_result, y_expected)
print("SUCCESS!")
