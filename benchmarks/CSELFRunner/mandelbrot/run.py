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


# mostly copied from GEMV

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

rows = 16
cols = 16

# Derive ISL maps for inputs and outputs

oport = f"{{ R[i=0:{rows - 1}, j=0:{cols - 1}, k=0:2] -> [PE[4, i // 4] -> index[i,j,k]] }}"

output_color = 2

elf_paths = glob(f"{name}/bin/out_[0-9]*.elf")

runner = CSELFRunner(elf_paths, cmaddr=args.cmaddr)

runner.add_output_tensor(output_color, oport, dtype=np.float32)

# Simulate
runner.connect_and_run()

# Read tensor from simulation and compare it with the expected values
R = runner.out_tensor_dict["R"]

iters = R[:, :, 2]
# A simple in-terminal representation of the Mandelbrot set
print(iters)

ref = np.zeros((rows, cols))

x_lo, x_hi = -2.0, 1.0
y_lo, y_hi = -1.5, 1.5
max_iters = 32

for r, c in product(range(rows), range(cols)):
  x = c * (x_hi - x_lo) / (cols - 1) + x_lo
  y = r * (y_hi - y_lo) / (rows - 1) + y_lo

  val = np.csingle(x + y * 1j)

  for i in range(max_iters + 1):
    if abs(val) >= 2.0:
      break
    val = val * val + (x + y * 1j)
  ref[r, c] = i

np.testing.assert_equal(ref, iters)

print("SUCCESS")
