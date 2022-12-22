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
X_cols = 1

B_rows = matrixHeight
B_cols = 1

# Use a deterministic seed so that CI results are predictable
np.random.seed(seed=7)

A = np.random.rand(A_rows, A_cols).astype(np.float16)
X = np.random.rand(X_rows, X_cols).astype(np.float16)
B = np.random.rand(B_rows, B_cols).astype(np.float16)

# Compute expected result
expected = (A @ X) + B

# Use input argument for fabric offsets, otherwise assume (1,1)
fabric_offsets = args.fabric_offsets
if fabric_offsets:
  fabric_offset_col, fabric_offset_row = map(int, fabric_offsets.split(','))
else:
  fabric_offset_col = 1
  fabric_offset_row = 1

fab_offsets = fabric_offset_col, fabric_offset_row

# Program runs on a 4x4 rectangle of PEs
kernel_rows = 4
kernel_cols = 4

# Parse the compile metadata
with open(f"{name}/out.json", encoding="utf-8") as json_file:
  compile_data = json.load(json_file)

compile_colors = compile_data["colors"]
color_x = int(compile_colors["x_in"])
color_b = int(compile_colors["b_in"])
color_y = int(compile_colors["y_out"])
color_sentinel = int(compile_colors["sentinel"])

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

runner.set_symbol_rect("A", A_per_pe, offset=fab_offsets)

# Specify tensors to send to and extract from the simulation
# The input array which is streamed in is reshaped appropriately
# to fall into target PEs.
X_per_pe_rows = X_rows // kernel_rows
runner.add_input_array("X", color_x, "N",
                       np.reshape(X, (kernel_rows, X_per_pe_rows)), sentinel=color_sentinel)

B_per_pe_rows = B_rows // kernel_rows
runner.add_input_array("B", color_b, "W",
                       np.reshape(B, (kernel_rows, B_per_pe_rows)))

y_arr = np.zeros((kernel_rows, B_per_pe_rows), dtype=np.float16)
runner.add_output_array("y", color_y, "E", y_arr)

# Simulate
runner.connect_and_run()

# Read tensor from simulation and compare it with the expected values
result_tensor = runner.out_tensor_dict["y"]
result_tensor = np.reshape(result_tensor, expected.shape)

np.testing.assert_allclose(result_tensor, expected, atol=0.01, rtol=0)
print("SUCCESS!")
