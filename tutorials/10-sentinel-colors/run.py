#!/usr/bin/env cs_python

import argparse
from glob import glob
import numpy as np
from cerebras.elf.cs_elf_runner import CSELFRunner

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='the test name')
parser.add_argument("--cmaddr", help="IP:port for CS system")
args = parser.parse_args()
name = args.name

size = 10

elf_paths = glob(f"{name}/bin/out_[0-9]*.elf")

# Use a deterministic seed so that CI results are predictable
np.random.seed(seed=7)

# Setup a {size}x11 input tensor that is reduced along the second dimension
input_tensor = np.random.rand(size, 11).astype(np.float16)
expected = np.sum(input_tensor, axis=1)

runner = CSELFRunner(elf_paths, cmaddr=args.cmaddr)

# Distribute the input tensor uniformly along the west edge of the kernel
main_color_value = 0
trigger_result = 43

# ISL map to indicate the PE that will receive the input wavelet, along with the
# direction of the input wavelet
input_port_map = "{input[peIdx=0:9, valIdx=0:10] -> [PE[-1,peIdx] -> index[peIdx, valIdx]]}"
runner.add_input_tensor(main_color_value, input_port_map, input_tensor,
                        sentinel=trigger_result)

# Color along which we expect the output message
output_color = 1

# ISL map to indicate the PE that will produce the output wavelet, along with
# the direction of the output wavelet
max_idx = input_tensor.shape[0] - 1
output_port_map = f"{{out_tensor[idx=0:{max_idx}] -> [PE[1,idx] -> index[idx]]}}"
runner.add_output_tensor(output_color, output_port_map, np.float16)

# Proceed with simulation; fetch the output wavelets once simulation completes
runner.connect_and_run()
result_tensor = runner.out_tensor_dict["out_tensor"]

# Ensure that the result matches our expectation
np.testing.assert_allclose(result_tensor, expected, atol=0.05, rtol=0)
print("SUCCESS!")
