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

result_color = 2
result_port_map = f"{{result_out[peIdx=0:3,valIdx=0:2] -> [PE[peIdx,-1] -> index[peIdx,valIdx]]}}"
runner.add_output_tensor(result_color, result_port_map, np.float16)

runner.connect_and_run(sim_out_path)
result = runner.out_tensor_dict["result_out"].reshape(12)

oracle = [5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5]
np.testing.assert_allclose(result, oracle, atol=0.0001, rtol=0)
print("SUCCESS!")
