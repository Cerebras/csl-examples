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
from glob import glob
import numpy as np

from cerebras.elf.cs_elf_runner import CSELFRunner

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='the test name')
parser.add_argument("--cmaddr", help="IP:port for CS system")
args = parser.parse_args()
name = args.name

# Path to ELF and simulation output files
elf_paths = glob(f"{name}/bin/out_[0-9]*.elf")
sim_out_path = f"{name}/bin/core.out"

# Simulate ELF file and produce the simulation output
runner = CSELFRunner(elf_paths, cmaddr=args.cmaddr)

output_color = 2
output_port_map = f"{{out_tensor[idx=0:0] -> [PE[1,-1] -> index[idx]]}}"
runner.add_output_tensor(output_color, output_port_map, np.int16)

runner.connect_and_run(sim_out_path)

result_top = runner.get_symbol(2, 1, "global", np.uint16)
result_left = runner.get_symbol(1, 2, "global", np.uint16)
result_right = runner.get_symbol(3, 2, "global", np.uint16)
result_bottom = runner.get_symbol(2, 3, "global", np.uint16)

np.testing.assert_allclose(result_top, 0xdd)
np.testing.assert_allclose(result_left, 0xaa)
np.testing.assert_allclose(result_right, 0xbb)
np.testing.assert_allclose(result_bottom, 0xcc)
print("SUCCESS!")
