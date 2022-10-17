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


# The code for this 3D 25-point stencil was inspired by the proprietary code
# of TotalEnergies EP Research & Technology US.

import argparse
from glob import glob
import numpy as np
from cerebras.elf.cs_elf_runner import CSELFRunner

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='the test name')
parser.add_argument('--zDim', type=int, help='size of the Z dimension')
parser.add_argument('--size', type=int, help='size of the fabric in x and y dims')
parser.add_argument('--iterations', type=int, help='number of timesteps to simulate')
parser.add_argument('--out_color', type=int, help='color on which to output tally')
parser.add_argument('--tsc_color', type=int, help='color on which to output timestamp counter')
parser.add_argument('--iter_color', type=int, help='color on which to receive iteration count')
parser.add_argument('--cmaddr', help='IP:port for CS system')

args = parser.parse_args()
name = args.name
size = args.size
zDim = args.zDim
iterations = args.iterations
out_color = args.out_color
tsc_color = args.tsc_color
iter_color = args.iter_color

elf_paths = glob(f"{name}/bin/out_[0-9]*.elf")

runner = CSELFRunner(elf_paths, cmaddr=args.cmaddr)

iter_array = np.full(size, iterations, dtype=np.uint32)
input_port_map = f"{{input[idx=0:{size-1}] -> [PE[-1,idx] -> index[idx]]}}"
runner.add_input_tensor(iter_color, input_port_map, iter_array)

out_port_map = f"{{out_tensor[idx=0:0] -> [PE[{size-1},-1] -> index[idx]]}}"
runner.add_output_tensor(out_color, out_port_map, np.uint32)

tsc_port_map = f"{{tsc_tensor[idx=0:5] -> [PE[{size-1},-1] -> index[idx]]}}"
runner.add_output_tensor(tsc_color, tsc_port_map, np.uint16)

# Start simulation and wait for the tally message
runner.connect_and_run()

rect = ((1, 1), (size, size))
minValues = runner.get_symbol_rect(rect, "minValue", np.float32)
maxValues = runner.get_symbol_rect(rect, "maxValue", np.float32)
computedMax = maxValues.max()
computedMin = minValues.min()

print(f"[computed] min: {computedMin}, max: {computedMax}")

np.testing.assert_allclose(computedMin, -1.3100899, atol=0.01, rtol=0)
np.testing.assert_allclose(computedMax, 1200.9414062, atol=0.01, rtol=0)

def make_u48(words):
  return words[0] + (words[1] << 16) + (words[2] << 32)

def sub_ts(words):
  return make_u48(words[3:]) - make_u48(words[0:3])

cycles = sub_ts(runner.out_tensor_dict["tsc_tensor"])
cycles_per_element = cycles / (iterations * zDim)
print(f"cycles per element = {cycles_per_element}")
