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
from glob import glob

import numpy as np
from cerebras.elf.cs_elf_runner import CSELFRunner

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="the test name")
parser.add_argument("--cmaddr", help="IP:port for CS system")
parser.add_argument("--fabric-offsets", help="column,row offset into the fabric")
args = parser.parse_args()
name = args.name

# Create tensors for A, X, B.
matrixWidth = 16
matrixHeight = 32

A_rows = matrixHeight
A_cols = matrixWidth

X_rows = matrixWidth

B_rows = matrixHeight

# Use a deterministic seed so that CI results are predictable
np.random.seed(seed=7)

A = np.random.rand(A_rows, A_cols).astype(np.float32)
X = np.random.rand(X_rows).astype(np.float32)
B = np.random.rand(B_rows).astype(np.float32)

# Compute expected result
y_expected = (A @ X) + B

# Use input argument for fabric offsets, otherwise assume (0,0)
fabric_offsets = args.fabric_offsets
if fabric_offsets:
  fabric_offset_col, fabric_offset_row = map(int, fabric_offsets.split(','))
else:
  fabric_offset_col = 0
  fabric_offset_row = 0

fab_offsets = fabric_offset_col, fabric_offset_row

# Program runs on a 4x4 rectangle of PEs
kernel_rows = 4
kernel_cols = 4

# Specify path to ELF files, set up runner
elf_paths = glob(f"{args.name}/bin/out_[0-9]*.elf")
runner = CSELFRunner(elf_paths, cmaddr=args.cmaddr)

# Split A tensor across PEs
# A[M, N] -> kernel_cols * kernel_rows * A[M // kernel_cols, N // kernel_rows]
# We need to swap kernel_cols and kernel_rows dimensions as 'set_symbol_rect' API
# takes column as first index and row as second index
per_pe_rows = A_rows // kernel_rows
per_pe_cols = A_cols // kernel_cols

# After tiling tensor A across PE grid, it will have
# shape = (kernel_rows, kernel_cols, per_pe_rows, per_pe_cols)
A_per_pe = A.reshape(kernel_rows, per_pe_rows,
                     kernel_cols, per_pe_cols).transpose(2, 0, 1, 3)

runner.set_symbol_rect("A_tile", A_per_pe, offset=fab_offsets)

# Place x and b on PE (0,0). They will be scattered with collective comms
runner.set_symbol(0, 0, "x_src", X)
runner.set_symbol(0, 0, "b_src", B)

# Run the simulation
runner.connect_and_run()

# Collect the result y from PE (3,3) and compare to expected
y = runner.get_symbol(kernel_cols - 1, kernel_rows - 1, "final_result", np.float32)

np.testing.assert_allclose(y, y_expected, atol=0.01, rtol=0)
print("SUCCESS")
