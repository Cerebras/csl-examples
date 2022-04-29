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

pe_count = 10

elf_paths = glob(f"{name}/bin/out_[0-9]*.elf")

runner = CSELFRunner(elf_paths, cmaddr=args.cmaddr)

# Use a deterministic seed so that CI results are predictable
np.random.seed(seed=7)

main_color_value = 0
max_idx = pe_count - 1

# Send each PE a single randomly-generated float16 value
input_tensor = np.random.rand(pe_count).astype(np.float16)

# ISL map to indicate the PE that will receive the input wavelet, along with the
# direction of the input wavelet
input_port_map = f"{{in_tensor[idx=0:{max_idx}] -> [PE[idx,-1] -> index[idx]]}}"
runner.add_input_tensor(main_color_value, input_port_map, input_tensor)

# Color along which we expect the output message
output_color = 1

# ISL map to indicate the PE that will produce the output wavelet, along with
# the direction of the output wavelet
output_port_map = f"{{out_tensor[idx=0:{max_idx}] -> [PE[idx,1] -> index[idx]]}}"
runner.add_output_tensor(output_color, output_port_map, np.float16)

# Procced with simulation; fetch the output wavelets once simulation completes
sim_out_path = f"{name}/bin/core.out"
runner.connect_and_run(sim_out_path)
result_tensor = runner.out_tensor_dict["out_tensor"]

expected = np.sqrt(input_tensor)
np.testing.assert_allclose(result_tensor, expected, atol=0.001, rtol=0)
print("SUCCESS!")
