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
P, Nt, Kt, Mt = 4, 14, 14, 14

N = Nt * P
K = Kt * P
M = Mt * P

A = np.arange(N * K, dtype=np.float32).reshape((N, K)) % 100
B = np.arange(K * M, dtype=np.float32).reshape((K, M)) % 100

# Split it up into tiles that can be mapped to each PE
A_tiles_xy = np.array([np.vsplit(s, P) for s in np.hsplit(A, P)])
B_tiles_xy = np.array([np.vsplit(s, P) for s in np.hsplit(B, P)])

# Write tiles to PEs
for px, py in product(range(P), range(P)):
  A_tile = A_tiles_xy[px, py]
  B_tile = B_tiles_xy[px, py]

  runner.set_symbol(px, py, "A_tile", A_tile)
  runner.set_symbol(px, py, "B_tile", B_tile)

# Run the simulation
runner.connect_and_run()

starts = np.zeros((P, P), dtype=int)
ends = np.zeros((P, P), dtype=int)
def parse_timestamp(words):
  return words[0] | (words[1] << 16) | (words[2] << 32)

# Collect the results
C_tiles = np.zeros((P, P, Nt, Mt), dtype=np.float32)
for px, py in product(range(P), range(P)):
  C_tile = runner.get_symbol(px, py, "C_tile", np.float32).reshape(Nt, Mt)
  C_tiles[px, py] = C_tile

  # Get timestamping information
  start = runner.get_symbol(px, py, "tsc_start_buffer", np.uint16)
  end = runner.get_symbol(px, py, "tsc_end_buffer", np.uint16)
  starts[px, py] = parse_timestamp(start)
  ends[px, py] = parse_timestamp(end)

print(f"Cycles = {ends.max() - starts.min()}")

# Reshape the tensor into the right shape
C_result = C_tiles.transpose(1, 2, 0, 3).reshape(N, M)

# Check the result
C_expected = np.dot(A, B)
np.testing.assert_equal(C_expected, C_result)

print("SUCCESS")
