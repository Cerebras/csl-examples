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

# Path to ELF and simulation output files
elf_paths = [f"{name}/bin/out_0_0.elf", f"{name}/bin/out_1_0.elf"]
sim_out_path = f"{name}/bin/core.out"

runner = CSELFRunner(elf_paths, cmaddr=args.cmaddr)

# Set the number of values to pass through the reflection kernel
size = 48

# Set up an input tensor to be sent along a specific color
color0 = 0
color1 = 1
color2 = 2

# Generate a random input tensor of the desired size
input_values = np.random.randint(256, size=size, dtype=np.int16)

# Specify the input port map.  Here, we indicate that all elements of the
# tensor should be sent to the PE #0
max_idx = input_values.shape[0] - 1
input_port_map = f"{{in_tensor[idx=0:{max_idx}] -> [PE[-1,0] -> index[idx]]}}"
runner.add_input_tensor(color0, input_port_map, input_values)

output_port_map = f"{{out_tensor[idx=0:{max_idx}] -> [PE[1,1] -> index[idx]]}}"
runner.add_output_tensor(color2, output_port_map, np.int16)

# Simulate ELF file and produce the simulation output
runner.connect_and_run(sim_out_path)
result_tensor = runner.out_tensor_dict["out_tensor"]

# Ensure that the streamed (fabric) tensor is as expected
np.testing.assert_equal(result_tensor, input_values * 4)
print("SUCCESS!")
