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
from cerebras.elf.cself import ELFMemory

Pw = 15
Ph = 15
chunk_size = 3

Nx = Pw*chunk_size
Ny = Ph*chunk_size

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='the test name')
parser.add_argument('--cmaddr', help='IP:port for CS system')
args = parser.parse_args()
name = args.name

# Path to ELF files
elf_paths = glob(f"{name}/bin/out_*.elf")

# Simulate ELF files
runner = CSELFRunner(elf_paths, cmaddr=args.cmaddr)

rect = (Pw, Ph)
broadcast_data = np.ones((1, Ph, Nx)).astype(np.float32)
runner.set_symbol_rect("broadcast_data", broadcast_data)

scatter_data = np.ones((Pw, 1, Ny)).astype(np.int32)
runner.set_symbol_rect("scatter_data", scatter_data)

runner.connect_and_run()

correct_broadcast_recv = np.ones(Nx).astype(np.float32)
rect = ((0, 0), (Pw, Ph))
broadcast_recvs = runner.get_symbol_rect(rect, "broadcast_recv", dtype=np.float32)
elfs = ELFMemory(*elf_paths)
for x, y in elfs.iter_coordinates():
  if x == 0:
    continue
  np.testing.assert_equal(broadcast_recvs[x, y], correct_broadcast_recv)

correct_faddh_result = np.full(Nx, 14, dtype=np.float32)
rect = ((0, 0), (1, Ph))
faddh_results = runner.get_symbol_rect(rect, "faddh_result", dtype=np.float32)
for y in range(Ph):
  np.testing.assert_equal(faddh_results[0, y], correct_faddh_result)

correct_gather_recv = np.ones(Ny).astype(np.int32)
rect = ((0, 0), (Pw, 1))
gather_recvs = runner.get_symbol_rect(rect, "gather_recv", dtype=np.int32)
for x in range(Pw):
  np.testing.assert_equal(gather_recvs[x, 0], correct_gather_recv)

print("SUCCESS")
