#!/usr/bin/env cs_python

import argparse
import numpy as np

from cerebras.elf.cs_elf_runner import CSELFRunner

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='the test name')
parser.add_argument("--cmaddr", help="IP:port for CS system")
parser.add_argument("--tolerance", type=float, help="tolerance for result")
args = parser.parse_args()
name = args.name

# Path to ELF files
elf_paths = [f"{name}/bin/out_0_0.elf"]

# Simulate ELF files
runner = CSELFRunner(elf_paths, cmaddr=args.cmaddr)

result_color = 1
result_port_map = "{result_out[idx=0:0] -> [PE[1,0] -> index[idx]]}"
runner.add_output_tensor(result_color, result_port_map, np.float32)

tsc_color = 2
tsc_port_map = "{tsc_out[idx=0:5] -> [PE[1,0] -> index[idx]]}"
runner.add_output_tensor(tsc_color, tsc_port_map, np.uint16)

runner.connect_and_run()
result = runner.out_tensor_dict["result_out"]

# Helper functions for computing the delta in the cycle count
def make_u48(words):
  return words[0] + (words[1] << 16) + (words[2] << 32)

def subtract_timestamps(words):
  return make_u48(words[3:]) - make_u48(words[0:3])

cycles = subtract_timestamps(runner.out_tensor_dict["tsc_out"])
print("cycle count:", cycles)

np.testing.assert_allclose(result, np.pi, atol=args.tolerance, rtol=0)
print("SUCCESS!")
