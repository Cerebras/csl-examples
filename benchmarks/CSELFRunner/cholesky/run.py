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


from itertools import product
import argparse
from glob import glob
import numpy as np
from cerebras.elf.cs_elf_runner import CSELFRunner

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="the test name")
parser.add_argument("--cmaddr", help="IP:port for CS system")
args = parser.parse_args()
name = args.name

elf_paths = glob(f"{name}/bin/*.elf")
runner = CSELFRunner(elf_paths, cmaddr=args.cmaddr)

# Initialize the input matrices
P, Nt = 10, 4
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

# Write tiles to PEs
for px, py in product(range(P), range(P)):
  if px > py:
    continue

  M_tile = M_tiles_xy[px, py]
  runner.set_symbol(px, py, "tile", M_tile)

# Run the simulation
runner.connect_and_run()

# collect results
result_tiles = np.zeros(M_tiles_xy.shape, dtype=M_tiles_xy.dtype)
for px, py in product(range(P), range(P)):
  if px > py:
    continue

  tile = runner.get_symbol(px, py, "tile", np.float32).reshape(Nt, Nt)
  result_tiles[px, py] = tile

# reassemble result
result = result_tiles.transpose(1, 2, 0, 3).reshape(N, N)

np.testing.assert_almost_equal(result, L, decimal=2)

print("SUCCESS")
