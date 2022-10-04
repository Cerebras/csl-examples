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

# Path to ELF files
elf_paths = glob(f"{name}/bin/out_[0-9]*.elf")

# Simulate ELF files
runner = CSELFRunner(elf_paths, cmaddr=args.cmaddr)

result_color = 2
result_port_map = "{result_out[peIdx=0:3,valIdx=0:2] -> [PE[peIdx,-1] -> index[peIdx,valIdx]]}"
runner.add_output_tensor(result_color, result_port_map, np.float16)

runner.connect_and_run()
result = runner.out_tensor_dict["result_out"].reshape(12)

oracle = [5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5]
np.testing.assert_allclose(result, oracle, atol=0.0001, rtol=0)
print("SUCCESS!")
