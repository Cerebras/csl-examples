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

# pylint: disable=line-too-long,too-many-function-args

import struct
import json
import os
import subprocess
import time
from typing import List
from glob import glob
from ic import computeGaussianSource
import numpy as np
from cmd_parser import parse_args
from cerebras.elf.cs_elf_runner import CSELFRunner

SIZE = 10
ZDIM = 10
PATTERN = 5
ITERATIONS = 10
DX = 20
arch_default = "wse2"
# "+5" for infrastructure of memcpy
# "+2" for a halo of size 1
FABRIC_WIDTH = SIZE + 2 + 5
FABRIC_HEIGHT = SIZE + 2

FILE_PATH = os.path.realpath(__file__)
MEMCPY_DIR = os.path.dirname(FILE_PATH)
DEPIPELINE_DIR = os.path.dirname(MEMCPY_DIR)
TEST_DIR = os.path.dirname(DEPIPELINE_DIR)
HPC_DIR = os.path.dirname(TEST_DIR)
ROOT_DIR = os.path.dirname(HPC_DIR)
CSL_DIR = os.path.join(ROOT_DIR, "cslang")
DRIVER = os.path.join(CSL_DIR, "build") + "/bin/cslc"



def float_to_hex(f):
  return hex(struct.unpack('<I', struct.pack('<f', f))[0])

def make_u48(words):
  return words[0] + (words[1] << 16) + (words[2] << 32)

def sub_ts(words):
  return make_u48(words[3:]) - make_u48(words[0:3])



def csl_compile(
    cslc: str,
    arch: str,
    width: int,
    height: int,
    core_fabric_offset_x: int, # fabric-offsets of the core
    core_fabric_offset_y: int,
    zDim: int,
    sourceLength: int,
    dx: int,
    srcX: int,
    srcY: int,
    srcZ: int,
    fabric_width: int,
    fabric_height: int,
    name: str,
    c_h2d: List[int],
    c_d2h: List[int]
)  -> List[str]:
  """Generate ELFs for the layout."""

  assert core_fabric_offset_x == 4
  assert core_fabric_offset_y == 1

  start = time.time()
  # CSL Compilation Step
  args = []
  args.append(cslc) # command
  args.append("code_memcpy.csl")
  args.append(f"--fabric-dims={fabric_width},{fabric_height}")
  args.append(f"--fabric-offsets={core_fabric_offset_x},{core_fabric_offset_y}")
  args.append(f"--params=width:{width},height:{height},zDim:{zDim},sourceLength:{sourceLength}")
  args.append(f"--params=dx:{dx}")
  args.append(f"--params=srcX:{srcX},srcY:{srcY},srcZ:{srcZ}")
  args.append(f"--arch={arch}")
  args.append("--verbose")
  args.append("--memcpy")
  args.append(f"-o={name}_code")
  num_h2d = len(c_h2d)
  for j in range(num_h2d):
    c = c_h2d[j]
    args.append(f"--params=MEMCPYH2D_DATA_{j+1}_ID:{c}")
  num_d2h = len(c_d2h)
  for j in range(num_d2h):
    c = c_d2h[j]
    args.append(f"--params=MEMCPYD2H_DATA_{j+1}_ID:{c}")
  print(f"subprocess.check_call(args = {args}")
  subprocess.check_call(args)

  end = time.time()
  print(f"Code compiled in {end-start}s")

  elf_paths = glob(f"{name}_code/bin/out_[0-9]*.elf")

  return elf_paths


def main():
  """Main method to run the example code."""

  args = parse_args()

  # Path to the CSLC driver
  cslc = DRIVER
  print(f"cslc = {cslc}")

  name = args.name
  dx = args.dx
  iterations = args.iterations

  source, sourceLength = computeGaussianSource(iterations)
  print("Gaussian source computed")
  print(f"sourceLength = {sourceLength}")
  print(f"source = {source}")

  if args.skip_compile:
    # Parse the compile metadata
    compile_data = None
    with open(f"{name}_code/out.json", encoding="utf-8") as json_file:
      compile_data = json.load(json_file)
    assert compile_data is not None

    size = int(compile_data["params"]["width"])
    zDim = int(compile_data["params"]["zDim"])
  else:
    size = args.size
    zDim = args.zDim

  width = size
  height = size

  fabric_offset_x = 1
  fabric_offset_y = 1

  # if WSE is the target, fabric_[width|height] must be the size of WSE
  if args.fabric_width is not None:
    fabric_width = args.fabric_width
  else:
    fabric_width = fabric_offset_x + width + 5 + 1

  if args.fabric_height is not None:
    fabric_height = args.fabric_height
  else:
    fabric_height = fabric_offset_y + height + 1

  print(f"width = {width}, height={height}")
  print(f"fabric_offset_x = {fabric_offset_x}, fabric_offset_y={fabric_offset_y}")
  print(f"fabric_width = {fabric_width}, fabric_height={fabric_height}")

  assert fabric_width >= (fabric_offset_x + width + 5 + 1)
  assert fabric_height >= (fabric_offset_y + height + 1)

  srcX = int(width / 2) - 5
  srcY = int(height / 2) - 5
  srcZ = int(zDim / 2) - 5
  assert srcX >= 0
  assert srcY >= 0
  assert srcZ >= 0
  print(f"srcX (x-coordinate of the source) = width/2 - 5  = {srcX}")
  print(f"srcY (y-coordinate of the source) = height/2 - 5 = {srcY}")
  print(f"srcZ (z-coordinate of the source) = zdim/2 - 5   = {srcZ}")

  MEMCPYH2D_DATA_1 = 0  # 1st H2D

  MEMCPYD2H_DATA_1 = 1  # 1st D2H
  MEMCPYD2H_DATA_2 = 2  # 2nd D2H

  c_h2d = []
  c_h2d.append(MEMCPYH2D_DATA_1)

  c_d2h = []
  c_d2h.append(MEMCPYD2H_DATA_1)
  c_d2h.append(MEMCPYD2H_DATA_2)

  print(f"c_h2d = {c_h2d}")
  print(f"c_d2h = {c_d2h}")

  if not args.skip_compile:
    print("Cleaned up existing elf files before compilation")
    elf_paths = glob(f"{name}_code_*.elf")
    for felf in elf_paths:
      os.remove(felf)

    core_fabric_offset_x = fabric_offset_x + 3
    core_fabric_offset_y = fabric_offset_y

    start = time.time()
    csl_compile(
        cslc, arch_default, width, height, core_fabric_offset_x, core_fabric_offset_y,
        zDim, sourceLength, dx, srcX, srcY, srcZ,
        fabric_width, fabric_height, name,
        c_h2d,
        c_d2h)
    end = time.time()
    print(f"compilation of kernel in {end-start}s")
  else:
    print("skip-compile: No compilation, read existing ELFs")

  if args.skip_run:
    print("skip-run: early return")
    return

#----------- run the test --------

  # vp[h][w][l] = 10.3703699112
  vp_all = 10.3703699112
  vp = np.full(width*height*zDim, vp_all, dtype=np.float32)
  vp = vp.reshape(height, width, zDim)

  # source_all[h][w][l]
  source_all = np.zeros(width*height*zDim).reshape(width*height*zDim, 1).astype(np.float32)
  for tidx in range(sourceLength):
    #source_all[(srcY, srcX, tidx, 1)] = source[tidx]
    offset = srcY * width*zDim + srcX * zDim + tidx
    source_all[offset] = source[tidx]
  source_all = source_all.reshape(height, width, zDim)

  # - iterations: f32
  # - vp : f32[zDim]
  # - source: f32[zDim]
  h2d_data = np.zeros(width*height*(2*zDim+1)).reshape(height, width, (2*zDim+1)).astype(np.float32)

  for h in range(height):
    for w in range(width):
      h2d_data[(h, w, 0)] = iterations
      for l in range(zDim):
        h2d_data[(h, w, l+1)] = vp[(h, w, l)]
      for l in range(zDim):
        h2d_data[(h, w, l+1+zDim)] = source_all[(h, w, l)]

# prepare the simulation
  elf_list = glob(f"{name}_code/bin/out_[0-9]*.elf")
  elfs_east = glob(f"{name}_code/east/bin/out_[0-9]*.elf")
  elfs_west = glob(f"{name}_code/west/bin/out_[0-9]*.elf")

  elf_list = elf_list + elfs_east + elfs_west
  print(f"elf_list = {elf_list}")

#
# Step 2: the user creates CSRunner with user's kernel image
#
  simulator = CSELFRunner(elf_list, debug_mode=True, cmaddr=args.cmaddr,\
        height=height, width=width, input_colors=set(c_h2d), output_colors=set(c_d2h))

#
# Step 3: The user has to specify H2D/D2H
#
  # H2D [h][w][2*zDim+1]
  iportmapA = f"{{ A[j=0:{height-1}][i=0:{width-1}][k=0:{2*zDim+1-1}] \
    -> [PE[i, j] -> index[k]] }}"
  # D2H [h][w][5]
  oportmap1 = f"{{ maxmin_time[j=0:{height-1}][i=0:{width-1}][k=0:{5-1}] \
    -> [PE[i, j] -> index[k]] }}"
  # D2H [h][w][zDim]
  oportmap2 = f"{{ z[j=0:{height-1}][i=0:{width-1}][k=0:{zDim-1}] -> [PE[i, j] -> index[k]] }}"

  simulator.add_input_tensor(MEMCPYH2D_DATA_1, iportmapA, h2d_data)
  simulator.add_output_tensor(MEMCPYD2H_DATA_1, oportmap1, np.float32)
  simulator.add_output_tensor(MEMCPYD2H_DATA_2, oportmap2, np.float32)

#
# Step 4: run HW/simfab
#
  start = time.time()
  simulator.connect_and_run()
  end = time.time()
  print(f"Run done in {end-start}s")

  #[USER] move the sim log
  if args.cmaddr is None:
    # move simulation log and core dump to the given folder
    sim_log = f"{name}_code/sim.log"
    mv_sim_cmd = f"mv sim.log {sim_log}"
    os.system(mv_sim_cmd)

    mv_simfab_traces_cmd = f"mv simfab_traces {name}_code"
    ret = os.system(mv_simfab_traces_cmd)
    err_msg = f"{name}_code/simfab_traces exists, please remove it first"
    assert ret == 0, err_msg

#
# step 5: verification
#
  maxmin_time_hwl = simulator.out_tensor_dict["maxmin_time"]
  z_hwl = simulator.out_tensor_dict["z"]

  # D2H(max/min)
  # d2h_buf_f32[0] = maxValue
  # d2h_buf_f32[1] = minValue
  # D2H (timestamps)
  # d2h_buf_f32[2] = {tscStartBuffer[1], tscStartBuffer[0]}
  # d2h_buf_f32[3] = {tscEndBuffer[0], tscStartBuffer[2]}
  # d2h_buf_f32[4] = {tscEndBuffer[2], tscEndBuffer[1]}
  maxValues_d2h = np.zeros(width*height).reshape(height, width).astype(np.float32)
  for h in range(height):
    for w in range(width):
      maxValues_d2h[(h, w)] = maxmin_time_hwl[(h, w, 0)]

  minValues_d2h = np.zeros(width*height).reshape(height, width).astype(np.float32)
  for h in range(height):
    for w in range(width):
      minValues_d2h[(h, w)] = maxmin_time_hwl[(h, w, 1)]

  computedMax = maxValues_d2h.max()
  computedMin = minValues_d2h.min()
  print(f"[computed] min_d2h: {computedMin}, max_d2h: {computedMax}")

  timestamp_d2h = np.zeros(width*height*6).reshape(width, height, 6).astype(np.uint16)
  for w in range(width):
    for h in range(height):
      hex_t0 = int(float_to_hex(maxmin_time_hwl[(h, w, 2)]), base=16)
      hex_t1 = int(float_to_hex(maxmin_time_hwl[(h, w, 3)]), base=16)
      hex_t2 = int(float_to_hex(maxmin_time_hwl[(h, w, 4)]), base=16)
      timestamp_d2h[(w, h, 0)] = hex_t0 & 0x0000ffff
      timestamp_d2h[(w, h, 1)] = (hex_t0 >> 16) & 0x0000ffff
      timestamp_d2h[(w, h, 2)] = hex_t1 & 0x0000ffff
      timestamp_d2h[(w, h, 3)] = (hex_t1 >> 16) & 0x0000ffff
      timestamp_d2h[(w, h, 4)] = hex_t2 & 0x0000ffff
      timestamp_d2h[(w, h, 5)] = (hex_t2 >> 16) & 0x0000ffff
  tsc_tensor_d2h = np.zeros(6).astype(np.uint16)
  tsc_tensor_d2h[0] = timestamp_d2h[(width-1, 0, 0)]
  tsc_tensor_d2h[1] = timestamp_d2h[(width-1, 0, 1)]
  tsc_tensor_d2h[2] = timestamp_d2h[(width-1, 0, 2)]
  tsc_tensor_d2h[3] = timestamp_d2h[(width-1, 0, 3)]
  tsc_tensor_d2h[4] = timestamp_d2h[(width-1, 0, 4)]
  tsc_tensor_d2h[5] = timestamp_d2h[(width-1, 0, 5)]

  print(f"tsc_tensor_d2h = {tsc_tensor_d2h}")
  cycles = sub_ts(tsc_tensor_d2h)
  cycles_per_element = cycles / (iterations * zDim)
  print(f"cycles per element = {cycles_per_element}")

  zMax_d2h = z_hwl.max()
  zMin_d2h = z_hwl.min()
  print(f"[computed] zMin_d2h: {zMin_d2h}, zMax_d2h: {zMax_d2h}")

  if zDim == 10 and size == 10 and iterations == 10:
    print("[verification] w=h=zdim=10, iters = 10, check golden vector")
    np.testing.assert_allclose(computedMin, -1.3100899, atol=0.01, rtol=0)
    np.testing.assert_allclose(computedMax, 1200.9414062, atol=0.01, rtol=0)
  elif zDim == 10 and size == 10 and iterations == 2:
    print("[verification] w=h=zdim=10, iters = 2, check golden vector")
    np.testing.assert_allclose(computedMin, -0.0939295, atol=0.01, rtol=0)
    np.testing.assert_allclose(computedMax, 57.403816, atol=0.01, rtol=0)
  else:
    print("Results are not checked for those parameters")
    assert False


if __name__ == "__main__":
  main()
