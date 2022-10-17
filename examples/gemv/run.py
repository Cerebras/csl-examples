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
parser.add_argument("--fabdims", help="width,height of fabric size")
args = parser.parse_args()
name = args.name

matrixWidth = 32
matrixHeight = 16

# Use a deterministic seed so that CI results are predictable
np.random.seed(seed=7)

# Generate random data for input matrix and vectors
A = np.random.rand(matrixWidth, matrixHeight).astype(np.float16)
x = np.random.rand(matrixHeight, 1).astype(np.float16)
b = np.random.rand(matrixWidth, 1).astype(np.float16)

# Compute expected result
expected = (A @ x) + b

# Derive ISL maps for inputs and outputs

# iport_x is a map which describes how the input's tensor `x` is distributed
# across the edge of the PE-rectangle (kernel), i.e., this is a 1D map.
# `x` is divided equally across all 4 PEs on the edge. The i-th element of `x` is
# sent to PE[i//4, -1]. The -1 here is important as it describes that the
# elements are actually coming from above the kernel. I.e. they will be
# distributed across the NORTH edge of the rectangular kernel.
iport_x = "{ x[i=0:15, 0]-> [PE[i//4, -1]-> index[i mod 4]]}"

# Very similar to the above map for `x`, iport_b describes how `b` is sent to
# the rectangular kernel. Here the elements of `b` are chunked equally across 4
# PEs and element i goes to PE[-1, i//8]. The -1 in the first coordinate means
# that this input will be sent to the WEST edge of the kernel, in chunks of 8
# elements per PE.
iport_b = "{ b[i=0:31, 0]-> [PE[-1, i//8]-> index[i mod 8]]}"

# oport_y describes how outputs are to be collected across an edge of the kernel.
# This map is quite similar to the iportmap for `b` in that the vector `y` is
# distributed vertically across an edge of the kernel. However, the edge is
# now the EAST edge instead of the WEST one. This is indicated by the i-th
# element of `y` mapping to PE[4, i//8] - since the width of the kernel
# is 4, PE[4, i//8] lies to the right or EAST of the kernel indicating the
# kernel will output `y` across its EAST edge.
oport_y = "{ y[i=0:31, 0]-> [PE[4, i//8]-> index[i mod 8]]}"

fabdims = args.fabdims
if fabdims:
  fabric_width, fabric_height = map(int, fabdims.split(','))
else:
  fabric_width = 6
  fabric_height = 6

kernel_width = 4
kernel_height = 4

# Parse the compile metadata
compile_data = None
with open(f"{name}/out.json", encoding="utf-8") as json_file:
  compile_data = json.load(json_file)
assert compile_data is not None
compile_colors = compile_data["colors"]
color_x = int(compile_colors["x_in"])
color_b = int(compile_colors["b_in"])
color_y = int(compile_colors["y_out"])
color_sentinel = int(compile_colors["sentinel"])

elf_paths = glob(f"{args.name}/bin/out_[0-9]*.elf")

runner = CSELFRunner(elf_paths, cmaddr=args.cmaddr)

per_pe_cols = matrixWidth // kernel_width
per_pe_rows = matrixHeight // kernel_height
shape = (kernel_width, kernel_height, per_pe_cols, per_pe_rows)
A_per_pe = np.zeros(shape, np.float16)
for i in range(kernel_width):
  i_offset = i * per_pe_rows
  for j in range(kernel_height):
    j_offset = j * per_pe_cols
    A_per_pe[i][j] = A[j_offset:j_offset+per_pe_cols, i_offset:i_offset+per_pe_rows]

runner.set_symbol_rect("A", A_per_pe, offset=(1, 1))

# Specify tensors to send to and extract from the simulation
runner.add_input_tensor(color_x, iport_x, x, sentinel=color_sentinel)
runner.add_input_tensor(color_b, iport_b, b)
runner.add_output_tensor(color_y, oport_y, dtype=np.float16)

# Simulate
runner.connect_and_run()

# Read tensor from simulation and compare it with the expected values
result_tensor = runner.out_tensor_dict["y"]

np.testing.assert_allclose(result_tensor, expected, atol=0.01, rtol=0)
print("SUCCESS!")
