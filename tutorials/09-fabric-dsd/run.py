#!/usr/bin/env cs_python

import argparse
import numpy as np
from cerebras.elf.cs_elf_runner import CSELFRunner

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='the test name')
parser.add_argument("--cmaddr", help="IP:port for CS system")
args = parser.parse_args()
name = args.name

# Path to ELF files
elf_paths = [f"{name}/bin/out_0_0.elf"]

runner = CSELFRunner(elf_paths, cmaddr=args.cmaddr)

# Size of the input and output tensors; use this value when compiling the
# program, e.g. `cslc --params=size:64 --fabric-dims=3,3 --fabric-offsets=1,1`
size = 64

# Input and output colors
inColor = 1
outColor = 2

# Generate a random input tensor of the desired size
input_tensor = np.random.randint(256, size=size, dtype=np.int16)

# Set up an input tensor to be sent along a specific color
max_idx = input_tensor.shape[0] - 1
input_port_map = f"{{in_tensor[idx=0:{max_idx}] -> [PE[-1,0] -> index[idx]]}}"
runner.add_input_tensor(inColor, input_port_map, input_tensor)

# Set up an output tensor to receive along a specific color
output_port_map = f"{{out_tensor[idx=0:{max_idx}] -> [PE[1,0] -> index[idx]]}}"
runner.add_output_tensor(outColor, output_port_map, np.int16)

# Run on the simulator
runner.connect_and_run()

# Ensure that the result matches expectations
result_tensor = runner.out_tensor_dict["out_tensor"]
np.testing.assert_equal(result_tensor, input_tensor + 1)
print("SUCCESS!")
