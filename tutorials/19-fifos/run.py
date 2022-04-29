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
import json
from glob import glob
import numpy as np

from cerebras.elf.cs_elf_runner import CSELFRunner

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='the test name')
parser.add_argument("--cmaddr", help="IP:port for CS system")
args = parser.parse_args()
name = args.name

# Path to ELF and simulation output files
elf_paths = glob(f"{name}/bin/out_*.elf")
sim_out_path = f"{name}/bin/core.out"

# Simulate ELF file and produce the simulation output
runner = CSELFRunner(elf_paths, cmaddr=args.cmaddr)

np.random.seed(seed=7)

# Parse the compile metadata
compile_data = None
with open(f"{name}/out.json") as json_file:
  compile_data = json.load(json_file)
assert compile_data is not None
compile_params = compile_data["params"]
size = int(compile_params["num_elements_to_process"])
input_tensor = np.random.random(size).astype(np.float16)
input_port_map = f"{{in_tensor[idx=0:{size - 1}] -> [PE[-1,0] -> index[idx]]}}"
inColor = 0 # The input color from which the kernel will read data.
runner.add_input_tensor(inColor, input_port_map, input_tensor)

output_port_map = f"{{out_tensor[idx=0:{size - 1}] -> [PE[1,0] -> index[idx]]}}"
output_color = 3 # The output color to which the kernel will send data.
runner.add_output_tensor(output_color, output_port_map, np.float16)

runner.connect_and_run(sim_out_path)

result_tensor = runner.out_tensor_dict["out_tensor"]

add_ten_negate = -(input_tensor + 10.0)
expected = add_ten_negate * add_ten_negate * add_ten_negate

np.testing.assert_equal(result_tensor, expected)
print("SUCCESS!")
