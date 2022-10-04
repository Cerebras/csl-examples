#!/usr/bin/env cs_python

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
sim_out_path = f"{name}/bin/core.out"

runner = CSELFRunner(elf_paths, cmaddr=args.cmaddr)

runner.add_output_tensor(output_color, oport, dtype=np.float32)

# Simulate and write the simulation output to `core.out`
runner.connect_and_run(sim_out_path)

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
