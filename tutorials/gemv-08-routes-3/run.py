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

np.random.seed(seed=7)

# Construct A, x, b
#
# In the previous examples, we used arange to initialize
# the elements of A. This example can support any number
# of PEs, and thus the matrix A could be quite large, so
# arange is no longer suitable. Instead, we use rand() to
# initialize the elements to random numbers between 0 and
# 1, to avoid numerical issues in our calculation."
#
# The upper bound of error estimate is proportional to
# (|A|*|x|+|b|). If the magnitude of A is large, any small
# mistake could be invisible. For example, if one entry is
# not calculated correctly, it may still pass because the
# error bound is too big. So rand() is better than arange().
A = np.random.rand(M, N).astype(np.float32)
x = np.full(shape=N, fill_value=1.0, dtype=np.float32)
b = np.full(shape=M, fill_value=2.0, dtype=np.float32)

# Calculate expected y
y_expected = A@x + b

# Size of N dimension on each PE
N_per_PE = N // kernel_x_dim
M_per_PE = M // kernel_y_dim

# Construct a runner using SdkRuntime
runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

# Get symbols for A, x, y on device
A_symbol = runner.get_id('A')
x_symbol = runner.get_id('x')
y_symbol = runner.get_id('y')

# Load and run the program
runner.load()
runner.run()

# Copy b into y of PEs (0, 0) to (0, kernel_y_dim-1)
runner.memcpy_h2d(y_symbol, b, 0, 0, 1, kernel_y_dim, M_per_PE, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# Copy chunks of A into all PEs
# Each chunk on each PE is stored column major
A_prepared = A.reshape(kernel_y_dim, M_per_PE, kernel_x_dim, N_per_PE).transpose(0, 2, 3, 1).ravel()
runner.memcpy_h2d(A_symbol, A_prepared, 0, 0, kernel_x_dim, kernel_y_dim, M_per_PE*N_per_PE,
  streaming=False, order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT,
  nonblock=False)

# Copy x into PEs (0, 0) and (kernel_x_dim-1, 0)
# PE (0, 0) gets first N/2 elements; PE (1, 0) gets last N/2 elements
runner.memcpy_h2d(x_symbol, x, 0, 0, kernel_x_dim, 1, N_per_PE, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# Launch the compute function on device
runner.launch('compute', nonblock=False)

# Copy y back from PEs (kernel_x_dim-1, 0) and (kernel_x_dim-1, kernel_y_dim-1)
y_result = np.zeros([M], dtype=np.float32)
runner.memcpy_d2h(y_result, y_symbol, kernel_x_dim-1, 0, 1, kernel_y_dim, M_per_PE, streaming=False,
  order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)

# Stop the program
runner.stop()

# Ensure that the result matches our expectation
# The formula of assert_allclose() is
#   |a-b| <= atol + rtol * |b|
# where rtol = 1.e-5, atol=1.e-8
#
# However the magnitude of y_result or y_expected is (|A|*|x| + |b|), so
# it is likely to fail the absolute tolerance (atol) when |A| is large.
#
# Using the norm-wise error is perferred for large dimension.
#   |y_result - y_expected|/(|A|*|x| + |b|) < tol
#
r = y_result - y_expected
nrm_r = np.linalg.norm(r, np.inf)
nrm_x = np.linalg.norm(x, np.inf)
nrm_b = np.linalg.norm(b, np.inf)
nrm_A = np.linalg.norm(A, np.inf)
print(f"|y_result - y_expected| = {nrm_r}")
print(f"|x| = {nrm_x}")
print(f"|b| = {nrm_b}")
print(f"|A| = {nrm_A}")

relerr = nrm_r/(nrm_A*nrm_x + nrm_b)
print(f"|y_result - y_expected|/(|A|*|x| + |b|) = {relerr}")

assert relerr < 1.e-6, "norm-wise relative error is too big, something must be wrong"

# The norm-wise estimate sometimes over-estimates too much, we can also
# check the absolute error when the matrix is small.
#
# too large. If it fails (but norm-wise estimate can pass), try
#   np.testing.assert_allclose(y_result/nrm_A, y_expected/nrm_A)
if N < 10:
  np.testing.assert_allclose(y_result, y_expected)

print("SUCCESS!")
