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

# 1D:
# 2D:

import argparse
import json
import tempfile
from glob import glob
import numpy as np
from cerebras.elf.cs_elf_runner import CSELFRunner

def arguments():
  parser = argparse.ArgumentParser(description="FFT parameters.")
  parser.add_argument(
      "-no_check",
      "--no_check_outcome",
      action="store_true",
      help="Do not validate outcome against NumPy implementation",
  )
  parser.add_argument(
      "-v",
      "--verbose",
      action="store_true",
      help="Add verbose output, with sim log files per PE.",
  )
  parser.add_argument(
      "-n",
      "--name",
      required=True,
      default="out",
      help="Output/input file name prefix.",
  )
  parser.add_argument("--fabric-offsets", help="column,row of kernel offset")
  parser.add_argument("--cmaddr", help="IP:port for CS system")
  return parser.parse_args()

args = arguments()

# Parse the compile metadata
compile_data = None
with open(f"{args.name}/out.json") as json_file:
  compile_data = json.load(json_file)
assert compile_data is not None
compile_colors = compile_data["colors"]
compile_params = compile_data["params"]
output_color = int(compile_colors["output_color"])

DIM = int(compile_params["DIM"])
Nz = int(compile_params["Nz"])
FP = int(compile_params["FP"])
assert DIM in [1, 2], "only 1D and 2D supported"
assert (DIM == 1 and Nz >= 2) or (DIM == 2 and Nz >= 4), \
  f"Minimum problem size Nz for DIM={DIM} is not met"
assert Nz & (Nz-1) == 0, "Nz must be a power of 2"
assert FP in [1, 2]
assert FP == 1 or DIM == 1, "2D does not support FP2"

offset_col, offset_row = map(int, args.fabric_offsets.split(','))
fabric_offsets = offset_col, offset_row

width = 1 if DIM == 1 else Nz
is_2D = (width > 1)
Nx = Nz if is_2D else width
height = 1
Ny = height
precision_type = np.float16 if FP == 1 else np.float32

Nh = Nz >> 1
Tx = int(Nx / width)
Ty = int(Ny / height)
Tz = Tx

print(f"Nx:{Nx}, Ny:{Ny}, Nz:{Nz}, width:{width}, height:{height}, 2D:{is_2D}")
print(f"Tx:{Tx}, Ty:{Ty}, Tz:{Tz}")

ELEM_SIZE = 2
BRICK_ELEM = Tx * Ty * Tz
BRICK_LEN = BRICK_ELEM * ELEM_SIZE
depth = int(Nz/Tz)
LOCAL_TENSOR_ELEM = BRICK_ELEM * depth
LOCAL_TENSOR_LEN = LOCAL_TENSOR_ELEM * ELEM_SIZE

GLOBAL_TENSOR_ELEM = Nx * Ny * Nz
GLOBAL_TENSOR_LEN = GLOBAL_TENSOR_ELEM * ELEM_SIZE

DEBUG = 1 if args.verbose else 0
CHECK_RES = 1 if not args.no_check_outcome else 0

f = np.zeros(Nz, dtype=precision_type)
exponent = np.pi * np.arange(Nh) / Nh
fI = np.sin(exponent).astype(precision_type)
fR = np.cos(exponent).astype(precision_type)
f[0::2] = fR
f[1::2] = fI

# Set the seed so that CI results are deterministic
np.random.seed(seed=7)

# Create Random array if CHECK_RES set, or a fixed array if not
if CHECK_RES == 1:
  random_array = np.random.random(GLOBAL_TENSOR_ELEM).astype(precision_type)
else:
  random_array = np.arange(GLOBAL_TENSOR_ELEM).astype(precision_type)

X_pre = random_array.reshape((Nx, Ny, Nz))
X = np.zeros((width, height, LOCAL_TENSOR_LEN), dtype=precision_type)

for x in range(Nx):
  for y in range(Ny):
    for z in range(Nz):
      offset = int((x%Tx))
      offset = offset + z*Tz
      X[int(x/Tx)][y][offset*2] = X_pre[x][y][z]
      X[int(x/Tx)][y][offset*2+1] = 0

print(f"X.shape: {X.shape}, X.dtype = {X.dtype}")
print(f"X = {X}")

#########################################
elf_paths = glob(f"{args.name}/bin/out_[0-9]*.elf")
sim_out_path = f"{args.name}/bin/core.out"

# Write the ELF file and simulation output to a temporary directory
with tempfile.TemporaryDirectory() as dirpath:
  # ISL map to indicate the PE that will produce the output wavelet, along with
  # the direction of the output wavelet
  output_port_map = f"{{out_tensor[idx=0:0] -> [PE[-1,0] -> index[idx]]}}"

  runner = CSELFRunner(elf_paths, cmaddr=args.cmaddr)
  runner.add_output_tensor(output_color, output_port_map, np.int16)
  runner.set_symbol_rect("X", X, offset=fabric_offsets)

  # Write 'f' to every PE.
  f_fill = np.full((width, height, Nz), f, dtype=precision_type)
  runner.set_symbol_rect("f", f_fill, offset=fabric_offsets)

  # Run the computation on a simulated wafer or "simfabric"
  runner.connect_and_run(sim_out_path)

  rect = (fabric_offsets, (width, height))
  X_res = runner.get_symbol_rect(rect, "X", precision_type)

# Create result arrays
result_array_pre = np.zeros((height, width, LOCAL_TENSOR_ELEM), dtype=complex)
result_array = np.zeros((Ny, Nx, Nz), dtype=complex)
result_array_transposed = np.zeros((Ny, Nx, Nz), dtype=complex)

# Create Complex array out of Real and Imaginary parts
for row in range(height):
  row_res = X_res[0 : width, row, :]
  for i in range(width):
    for j in range(LOCAL_TENSOR_ELEM):
      result_array_pre[row][i][j] = complex(row_res[i][j * 2], row_res[i][j * 2 + 1])

for y in range(Ny):
  for x in range(Nx):
    for z in range(Nz):
      offset = int((x%Tx))
      offset = offset + z*Tz
      result_array[y][x][z] = result_array_pre[y][int(x/Tx)][offset]

# Transpose the array if needed and get the FFT result with the NumPy Reference
random_array_sq = random_array.reshape((Ny, Nx, Nz))
if is_2D:
  result_array_transposed[0] = result_array[0].T
  reference_array = np.fft.fft2(random_array_sq)
else:  # 1D FFT
  result_array_transposed = result_array
  reference_array = np.fft.fft(random_array_sq)

# Compare the Simfabric and NumPy results
print("\nResult array")
print(result_array_transposed)
print("\nRef array")
print(reference_array)

if CHECK_RES == 1:
  np.testing.assert_allclose(result_array_transposed, reference_array, rtol=0.25, atol=0)
  print("\nSUCCESS!")
