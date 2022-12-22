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
import sys
from glob import glob
import numpy as np

from cerebras.elf.cs_elf_runner import CSELFRunner

parser = argparse.ArgumentParser()
parser.add_argument('--name',
  help='Compile output directory of the test')
parser.add_argument('--size',
  help='Dimension along edge of tensor and program fabric rectangle', type=int)
parser.add_argument('--cmaddr',
  help='IP address of CS-2 when running on actual hardware')
parser.add_argument('--iters',
  help='Number of times to compute Hadamard products', type=int)

args = parser.parse_args()

name = args.name
cmaddr = args.cmaddr
size = args.size
iters = args.iters

# Number of iterations until value is sent to host along outColorB
report_count = 3

if iters < report_count:
  print("Program requires passing --iters 4 (or higher)")
  sys.exit()

# Path to ELF and simulation output files
elf_paths = glob(f"{name}/bin/out_[0-9]*.elf")
elf_paths += glob(f"{name}/west/bin/out_[0-9]*.elf")
elf_paths += glob(f"{name}/east/bin/out_[0-9]*.elf")
sim_out_path = "out-core.out"

# Input and output colors for batches
inColorA = 1
inColorB = 2
inColorC = 4

outColorA = 3
outColorB = 5

c_h2d = [inColorA, inColorB, inColorC]
c_d2h = [outColorA, outColorB]

runner = CSELFRunner(
    elf_paths,
    cmaddr,
    height=size,
    width=size,
    input_colors=c_h2d,
    output_colors=c_d2h,
)

# Generate a random input tensor of the desired size
np.random.seed(seed=7)

# Set up input tensor for tensor A used in computing Hadamard product
runner.add_input_array("A", inColorA, "W", None, reusable=True)

# Set up input tensor for tensor B used in computing Hadamard product
iport_mapB = f"{{in_tensorB[i=0:{size-1},j=0:{size-1},k=0:{size-1}] " \
              "-> [PE[i,j] -> index[k]]}}"
runner.add_input_tensor(inColorB, iport_mapB, None, reusable=True)

# Set up input tensor for tensor C used to perform alternative operation
iport_mapC = f"{{in_tensorC[i=0:{size-1},j=0:{size-1},k=0:0] " \
              "-> [PE[i,j] -> index[k]]}}"
runner.add_input_tensor(inColorC, iport_mapC, None, reusable=True)


# Set up output tensor for receiving Hadamard product along outColorA
oport_map = f"{{out_tensorA[i=0:{size-1},j=0:{size-1},k=0:{size-1}] " \
             "-> [PE[i,j] -> index[k]]}}"
runner.add_output_tensor(outColorA, oport_map, np.float32)

# Set up output tensor for receiving output along outColorB
# when report_count reaches 0 on each PE
out_tensor_B = np.zeros((size, size, 1)).astype(np.int32)
runner.add_output_array("out_tensorB", outColorB, "W", out_tensor_B,
  offset=0, reusable=True)


# Prepare the runner for selective batch mode
runner.load()
runner.start()

# Run initial batch
print("Running initial batch...")

# Prepare C tensor for input to device
in_tensorC = np.full((size, size, 1), report_count, np.uint16)
runner.prepare_input_tensor(inColorC, iport_mapC, in_tensorC)

# Run batch with input along inColorC, no output
# This will copy in_tensorC to the device
runner.run_batch([inColorC])

print("SUCCESS!")

for it in range(0, iters):
  print("Running batch iteration " + str(it) + "...")
  in_tensorA = np.random.rand(size, size, size).astype(np.float32)
  in_tensorB = np.random.rand(size, size, size).astype(np.float32)

  # Prepare A tensor for input to device
  # Axes must be swapped for correct mapping with input_array
  in_tensorA_shuffled = np.swapaxes(in_tensorA, 0, 1)
  runner.prepare_input_array(inColorA, 0, in_tensorA_shuffled)

  # Prepare B tensor for input to device
  runner.prepare_input_tensor(inColorB, iport_mapB, in_tensorB)

  # Run batch with inputs inColorA, inColor B and output outColorA
  # Read resulting output tensor out_tensor
  runner.run_batch([inColorA, inColorB, outColorA])
  result_tensorA = runner.out_tensor_dict["out_tensorA"]

  # Ensure that computed Hadamard product matches expected
  test_product = in_tensorA * in_tensorB
  np.testing.assert_equal(result_tensorA, test_product)

  print("SUCCESS!")

  if it == report_count - 1:
    print("Running additional batch on iteration " + str(it) + "...")
    # Run batch with output along outColorB, no input
    # This will copy values in iter_count back to host
    runner.run_batch([outColorB])
    result_tensorB = runner.out_tensor_dict["out_tensorB"]

    # Ensure that the result matches expectations
    # Result should be 2^(it + 1) - 1
    expectedB = np.full((size, size, 1), 2 ** (it + 1) - 1, np.uint16)
    result_tensorB = np.reshape(result_tensorB, expectedB.shape)
    np.testing.assert_equal(result_tensorB, expectedB)
    print("SUCCESS!")

runner.stop()
