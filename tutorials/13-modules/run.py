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
import numpy as np

from cerebras.elf.cs_elf_runner import CSELFRunner

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='the test name')
parser.add_argument("--cmaddr", help="IP:port for CS system")
args = parser.parse_args()
name = args.name

# Path to ELF files
elf_paths = [f"{name}/bin/out_0_0.elf"]

# Simulate ELF files
runner = CSELFRunner(elf_paths, cmaddr=args.cmaddr)

# Color along which we expect the output message
output_color = 1

# ISL map to indicate the PE that will produce the output wavelet, along with
# the direction of the output wavelet
max_idx = 9
output_port_map = f"{{out_tensor[idx=0:{max_idx}] -> [PE[1,0] -> index[idx]]}}"
runner.add_output_tensor(output_color, output_port_map, np.int16)

# Proceed with simulation; fetch the output wavelets once simulation completes
runner.connect_and_run()
result_tensor = runner.out_tensor_dict["out_tensor"]

expected = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
np.testing.assert_equal(result_tensor, expected)
print("SUCCESS!")
