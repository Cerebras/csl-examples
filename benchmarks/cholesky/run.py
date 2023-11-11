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


from itertools import product
import argparse
import json
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder # pylint: disable=no-name-in-module

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="the test name")
parser.add_argument("--cmaddr", help="IP:port for CS system")
args = parser.parse_args()
dirname = args.name

# Parse the compile metadata
with open(f"{dirname}/out.json", encoding="utf-8") as json_file:
  compile_data = json.load(json_file)
compile_params = compile_data["params"]
P = int(compile_params["P"])
Nt = int(compile_params["Nt"])
print(f"P = {P}, Nt = {Nt}")

print("WARNING: The simfab may take 90 sec")

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

sym_tile = runner.get_id("tile")

runner.load()
runner.run()

# Initialize the input matrices
N = P * Nt

counter = 1
L = np.zeros((N, N), dtype=np.float32)
for i in range(N):
  for j in range(i+1):
    L[i, j] = counter
    counter += 1

# M = LL^T except we only store the upper triangle
M = np.dot(L, L.T)
for i in range(N):
  for j in range(i+1, N):
    M[i, j] = 0

# Split it up into tiles that can be mapped to each PE
M_tiles_xy = np.array([np.vsplit(s, P) for s in np.hsplit(M, P)])

print("step 1: copy mode H2D prepares data in non-upper of A")
# Write tiles to PEs
for px, py in product(range(P), range(P)):
  if px > py:
    continue

  M_tile = M_tiles_xy[px, py]
  assert M_tile.size == Nt*Nt
  runner.memcpy_h2d(sym_tile, M_tile.ravel(), px, py, 1, 1, Nt*Nt, \
    streaming=False, data_type=memcpy_dtype, \
    order=MemcpyOrder.COL_MAJOR, nonblock=False)

print("stpe 2: call f_chol to compute A = L*L**T")
runner.launch("f_chol", nonblock=False)

print("step 3: copy mode D2H gather L")
# collect results
result_tiles = np.zeros(M_tiles_xy.shape, dtype=M_tiles_xy.dtype)
for px, py in product(range(P), range(P)):
  if px > py:
    continue

  tile = np.zeros(Nt*Nt, np.float32)
  runner.memcpy_d2h(tile, sym_tile, px, py, 1, 1, Nt*Nt,\
    streaming=False, data_type=memcpy_dtype, \
    order=MemcpyOrder.COL_MAJOR, nonblock=False)
  result_tiles[px, py] = tile.reshape(Nt, Nt)

runner.stop()

# reassemble result
result = result_tiles.transpose(1, 2, 0, 3).reshape(N, N)

np.testing.assert_almost_equal(result, L, decimal=2)

print("SUCCESS")
