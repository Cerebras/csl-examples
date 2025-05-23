#!/usr/bin/env python3

# Copyright 2025 Cerebras Systems.
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

""" compile the kernel
"""

import subprocess
from glob import glob
from typing import List, Optional

from cmd_parser import parse_args


def csl_compile_core(
    width: int,  # width of the core
    height: int,  # height of the core
    pe_length: int,
    file_config: str,
    comp_dir: str,
    fabric_width: int,
    fabric_height: int,
    core_fabric_offset_x: int,  # fabric-offsets of the core
    core_fabric_offset_y: int,
    arch: Optional[str],
    C0: int,
    C1: int,
    C2: int,
    C3: int,
    C4: int,
    channels: int,
) -> List[str]:
  """use cslc or sdk_debug_shell to compile the kernel"""

  cslc = "cslc"

  args = []
  args.append(cslc)  # command

  args.append(file_config)
  if arch is not None:
    args.append(f"--arch={arch}")
  args.append(f"--fabric-dims={fabric_width},{fabric_height}")
  args.append(f"--fabric-offsets={core_fabric_offset_x},{core_fabric_offset_y}")
  args.append(f"--params=width:{width},height:{height},pe_length:{pe_length}")
  args.append(f"--params=C0_ID:{C0}")
  args.append(f"--params=C1_ID:{C1}")
  args.append(f"--params=C2_ID:{C2}")
  args.append(f"--params=C3_ID:{C3}")
  args.append(f"--params=C4_ID:{C4}")
  args.append(f"-o={comp_dir}")
  args.append("--memcpy")
  args.append(f"--channels={channels}")
  args.append("--width-west-buf=0")
  args.append("--width-east-buf=0")

  print(f"subprocess.check_call(args = {args}")
  subprocess.check_call(args)

  elfs = glob(f"{comp_dir}/bin/out_[0-9]*.elf")

  return elfs


def main():
  """Main method to run the example code."""

  args, dirname = parse_args()

  height = args.m
  width = args.n
  pe_length = args.k
  channels = args.channels

  # prepare the simulation
  print("store ELFs and log files in the folder ", dirname)

  code_csl = "src/layout.csl"

  # "+5" is "demux adaptor" + "demux" + "cmd fan" + "mux" + "mux adaptor"
  # "+2" means halo of size 1
  min_fabric_width = width + 5 + 2
  min_fabric_height = height + 2

  core_fabric_offset_x = 4
  core_fabric_offset_y = 1

  fabric_width = 0
  fabric_height = 0
  if args.fabric_dims:
    w_str, h_str = args.fabric_dims.split(",")
    fabric_width = int(w_str)
    fabric_height = int(h_str)

  if fabric_width == 0 or fabric_height == 0:
    fabric_width = min_fabric_width
    fabric_height = min_fabric_height

  assert fabric_width >= min_fabric_width
  assert fabric_height >= min_fabric_height

  C0 = 0
  C1 = 1
  C2 = 2
  C3 = 3
  C4 = 4

  elf_list = csl_compile_core(
      width,
      height,
      pe_length,
      code_csl,
      dirname,
      fabric_width,
      fabric_height,
      core_fabric_offset_x,
      core_fabric_offset_y,
      args.arch,
      C0,
      C1,
      C2,
      C3,
      C4,
      channels,
  )

  if elf_list is None or len(elf_list) == 0:
    raise RuntimeError("Must have a non-empty list of ELFs to run")


if __name__ == "__main__":
  main()
