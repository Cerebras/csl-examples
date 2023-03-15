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

# Function for parsing trace output lives in csl_utils
from cerebras.elf.cs_elf_runner.lib import csl_utils

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='the test name')
parser.add_argument("--cmaddr", help="IP:port for CS system")
args = parser.parse_args()
name = args.name

# Path to ELF and simulation output files
elf_paths = glob(f"{name}/bin/out_[0-9]*.elf")
elf_list = elf_paths

# Parse the compile metadata
with open(f"{name}/out.json", encoding="utf-8") as json_file:
  compile_data = json.load(json_file)
  width = int(compile_data["params"]["width"])

# Simulate ELF file and produce the simulation output
runner = CSELFRunner(elf_list, cmaddr=args.cmaddr)

# Specify input color and input elements
red = 0
num_entries = 4
x = np.arange(num_entries, dtype=np.int16)

# Define ISL input port map for input element and add input tensor
iport = f"{{x[idx=0:{num_entries}-1]-> [PE[-1,0]-> index[idx]]}}"
runner.add_input_tensor(red, iport, x)

# Proceed with simulation
runner.connect_and_run()

result = np.zeros([width, num_entries])
for idx in range(width):
  # Get traces recorded in 'trace'
  trace_output = csl_utils.read_trace(runner, 1 + idx, 1, 'trace')

  # Copy all recorded trace values of variable 'global'
  result[idx,:]=trace_output[1::2]

  # Get timestamp traces recorded in 'times'
  timestamp_output = csl_utils.read_trace(runner, 1 + idx, 1, 'times')

  # Print out all traces for PE
  print("PE (", idx, ", 0): ")
  print("Trace: ", trace_output)
  print("Times: ", timestamp_output)
  print()

# In order, the host streams in 0, 1, 2, 3 from the West.
# Red tasks add values to running global sum on its PE.
# Blue tasks add 2*values to running global sum on its PE.
# Value of global var is recorded after each update.
# PEs 0, 2 activate blue task; 1, 3 activate red task.
# Trace values of global var on even PEs will be: 0, 2, 6, 12
# Trace values of global var on odd PEs will be: 0, 1, 3, 6
oracle = np.empty([width, num_entries])
for i in range(width):
  for j in range(num_entries):
    oracle[i, j] = ((i+1) % 2 + 1) * j * (j+1) / 2

# Assert that all trace values of 'global' are as expected
np.testing.assert_equal(result, oracle)
print("SUCCESS!")
