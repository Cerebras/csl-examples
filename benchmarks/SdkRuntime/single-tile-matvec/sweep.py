#!/usr/bin/env python

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


import subprocess
import argparse


def parse_args():
  """ parse the command line """

  parser = argparse.ArgumentParser(description="Sweep single tile matvec size parameter")
  parser.add_argument("--name", required=False, default="out",
                      help="Prefix of ELF files")
  parser.add_argument("--cmaddr", required=False, default="",
                      help="IP:port for CS system")
  parser.add_argument("--dims",
                      help="Fabric and program dimension, i.e. <W>,<H>")
  parser.add_argument("--iters", required=False, type=int, default=1,
                      help="Number of iterations for each matvec")

  args = parser.parse_args()

  return args


def cslc_compile(
    width: int,
    height: int,
    tile_size: int,
    iters: int,
    name: str
  ):
  """Generate ELFs for the layout"""

  args = []
  args.append("cslc") # command
  args.append(f"layout_matvec.csl") # file
  args.append(f"--fabric-dims={width+7},{height+2}") # options
  args.append("--fabric-offsets=4,1")
  args.append(f"--params=width:{width},height:{height},tile_size:{tile_size},iters:{iters}")

  args.append(f"-o={name}")
  args.append("--arch=wse2")
  args.append("--memcpy")
  args.append("--channels=1")
  print(f"subprocess.check_call(args = {args}")
  subprocess.check_call(args)

def cs_run(
    name: str,
    cmaddr: str
  ):
  """Run with cs_python"""

  args = []
  args.append("cs_python")
  args.append("run.py")
  args.append(f"--name={name}")
  args.append(f"--cmaddr={cmaddr}")
  subprocess.check_call(args)


def compile_and_run(
    width: int,
    height: int,
    tile_size: int,
    iters: int,
    name: str,
    cmaddr: str
  ):
  """Compile and run program."""

  cslc_compile(
    width,
    height,
    tile_size,
    iters,
    name)

  cs_run(name, cmaddr)


def main():
  """Main method to run the example code."""

  args = parse_args()

  w, h = args.dims.split(",")
  width = int(w)
  height = int(h)

  name = args.name # compilation output
  cmaddr = args.cmaddr
  iters = args.iters

  for tile_size in range(10,101,10):
    compile_and_run(
      width,
      height,
      tile_size,
      iters,
      name,
      cmaddr)


if __name__ == "__main__":
  main()
