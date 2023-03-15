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



import struct
import json
import os
import shutil
import subprocess
import time
from glob import glob
from pathlib import Path
from typing import List
from ic import computeGaussianSource
import numpy as np
from cmd_parser import parse_args

from cerebras.sdk.runtime import runtime_utils
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder

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


def cast_uint32(x):
  if isinstance(x, (np.float16, np.int16, np.uint16)):
    z = x.view(np.uint16)
    return np.uint32(z)
  if isinstance(x, (np.float32, np.int32, np.uint32)):
    return x.view(np.uint32)
  if isinstance(x, int):
    return np.uint32(x)
  if isinstance(x, float):
    z = np.float32(x)
    return z.view(np.uint32)

  raise RuntimeError(f"type of x {type(x)} is not supported")


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
    c_launch: int,
    n_channels: int,
    width_west_buf: int,
    width_east_buf: int
)  -> List[str]:
  """Generate ELFs for the layout."""

  start = time.time()
  # CSL Compilation Step
  args = []
  args.append(cslc)
  args.append("code_memcpy.csl")
  args.append(f"--fabric-dims={fabric_width},{fabric_height}")
  args.append(f"--fabric-offsets={core_fabric_offset_x},{core_fabric_offset_y}")
  args.append(f"--params=width:{width},height:{height},zDim:{zDim},sourceLength:{sourceLength}")
  args.append(f"--params=dx:{dx}")
  args.append(f"--params=srcX:{srcX},srcY:{srcY},srcZ:{srcZ}")
  args.append("--verbose")
  args.append(f"-o={name}_code")
  args.append(f"--params=LAUNCH_ID:{c_launch}")
  if arch is not None:
    args.append(f"--arch={arch}")
  args.append("--memcpy")
  args.append(f"--channels={n_channels}")
  args.append(f"--width-west-buf={width_west_buf}")
  args.append(f"--width-east-buf={width_east_buf}")
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

  n_channels = args.n_channels
  width_west_buf = args.width_west_buf
  width_east_buf = args.width_east_buf
  print(f"n_channels = {n_channels}")
  print(f"width_west_buf = {width_west_buf}, width_east_buf = {width_east_buf}")

  source, sourceLength = computeGaussianSource(iterations)
  print("Gaussian source computed")
  print(f"sourceLength = {sourceLength}")
  print(f"source = {source}")

  if args.skip_compile:
    # Parse the compile metadata
    with open(f"{name}_code/out.json", encoding="utf-8") as json_file:
      compile_data = json.load(json_file)

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
    fabric_width = fabric_offset_x + 3 + width + 2 + 1 + width_west_buf + width_east_buf

  if args.fabric_height is not None:
    fabric_height = args.fabric_height
  else:
    fabric_height = fabric_offset_y + height + 1

  print(f"width = {width}, height={height}")
  print(f"fabric_offset_x = {fabric_offset_x}, fabric_offset_y={fabric_offset_y}")
  print(f"fabric_width = {fabric_width}, fabric_height={fabric_height}")

  assert fabric_width >= (fabric_offset_x + width + 5 + 1 + width_west_buf + width_east_buf)
  assert fabric_height >= (fabric_offset_y + height + 1)

  srcX = width // 2 - 5
  srcY = height // 2 - 5
  srcZ = zDim // 2 - 5
  assert srcX >= 0
  assert srcY >= 0
  assert srcZ >= 0
  print(f"srcX (x-coordinate of the source) = width/2 - 5  = {srcX}")
  print(f"srcY (y-coordinate of the source) = height/2 - 5 = {srcY}")
  print(f"srcZ (z-coordinate of the source) = zdim/2 - 5   = {srcZ}")

  c_launch = 0

  print(f"c_launch = {c_launch}")

  if not args.skip_compile:
    print("Cleaned up existing elf files before compilation")
    elf_paths = glob(f"{name}_code_*.elf")
    for felf in elf_paths:
      os.remove(felf)

    core_fabric_offset_x = fabric_offset_x + 3 + width_west_buf
    core_fabric_offset_y = fabric_offset_y

    start = time.time()
    csl_compile(
        cslc, arch_default, width, height, core_fabric_offset_x, core_fabric_offset_y,
        zDim, sourceLength, dx, srcX, srcY, srcZ,
        fabric_width, fabric_height, name,
        c_launch,
        n_channels,
        width_west_buf,
        width_east_buf)
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

#
# Step 2: the user creates CSRunner
#
  memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
  memcpy_order = MemcpyOrder.ROW_MAJOR
  dirname = f"{name}_code"
  simulator = SdkRuntime(dirname, cmaddr=args.cmaddr)

  symbol_vp = simulator.get_id("vp")
  symbol_source = simulator.get_id("source")
  symbol_maxmin_time = simulator.get_id("maxmin_time")
  symbol_zout = simulator.get_id("zout")
  print(f"symbol_vp = {symbol_vp}")
  print(f"symbol_source = {symbol_source}")
  print(f"symbol_maxmin_time = {symbol_maxmin_time}")
  print(f"symbol_zout = {symbol_zout}")

  simulator.load()
  simulator.run()

  start = time.time()
#
# Step 3: The user has to prepare the sequence of H2D/D2H/RPC
#
  # H2D vp[h][w][zDim]
  iportmap_vp = f"{{ vp[j=0:{height-1}][i=0:{width-1}][k=0:{zDim-1}] \
    -> [PE[i, j] -> index[k]] }}"
  # H2D source[h][w][zDim]
  iportmap_source = f"{{ source[j=0:{height-1}][i=0:{width-1}][k=0:{zDim-1}] \
    -> [PE[i, j] -> index[k]] }}"

  # use the runtime_utils library to calculate memcpy args and shuffle data
  (px, py, w, h, l, data) = runtime_utils.convert_input_tensor(iportmap_vp, vp)
  simulator.memcpy_h2d(symbol_vp, data, px, py, w, h, l,
                       streaming=False, data_type=memcpy_dtype,
                       order=memcpy_order, nonblock=False)
  (px, py, w, h, l, data) = runtime_utils.convert_input_tensor(iportmap_source, source_all)
  simulator.memcpy_h2d(symbol_source, data, px, py, w, h, l,
                       streaming=False, data_type=memcpy_dtype,
                       order=memcpy_order, nonblock=False)

  # time marching: call f_activate_comp() to set num iters and start computation
  simulator.call("f_activate_comp", [cast_uint32(iterations)], nonblock=False)

  # D2H [h][w][5]
  oportmap1 = f"{{ maxmin_time[j=0:{height-1}][i=0:{width-1}][k=0:{5-1}] \
    -> [PE[i, j] -> index[k]] }}"
  # use the runtime_utils library to calculate memcpy args and manage output data
  (px, py, w, h, l, data) = runtime_utils.prepare_output_tensor(oportmap1, np.float32)
  simulator.memcpy_d2h(data, symbol_maxmin_time, px, py, w, h, l,
                       streaming=False, data_type=memcpy_dtype,
                       order=memcpy_order, nonblock=False)
  maxmin_time_hwl = runtime_utils.format_output_tensor(oportmap1, np.float32, data)

  # prepare zout: call f_prepare_zout()
  simulator.call("f_prepare_zout", [], nonblock=False)

  # D2H [h][w][zDim]
  oportmap2 = f"{{ z[j=0:{height-1}][i=0:{width-1}][k=0:{zDim-1}] -> [PE[i, j] -> index[k]] }}"
  (px, py, w, h, l, data) = runtime_utils.prepare_output_tensor(oportmap2, np.float32)
  simulator.memcpy_d2h(data, symbol_zout, px, py, w, h, l,
                       streaming=False, data_type=memcpy_dtype,
                       order=memcpy_order, nonblock=False)
  z_hwl = runtime_utils.format_output_tensor(oportmap2, np.float32, data)

  simulator.stop()
  end = time.time()

  print(f"Run done in {end-start}s")

  if args.cmaddr is None:
    # move simulation log and core dump to the given folder
    sim_log = f"{dirname}/sim.log"
    shutil.move("sim.log", sim_log)

    dst = Path(f"{dirname}/simfab_traces")
    if dst.exists():
      shutil.rmtree(dst)
    shutil.move("simfab_traces", dst)

#
# step 4: verification
#
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
    print("\nSUCCESS!")
  elif zDim == 10 and size == 10 and iterations == 2:
    print("[verification] w=h=zdim=10, iters = 2, check golden vector")
    np.testing.assert_allclose(computedMin, -0.0939295, atol=0.01, rtol=0)
    np.testing.assert_allclose(computedMax, 57.403816, atol=0.01, rtol=0)
    print("\nSUCCESS!")
  else:
    print("Results are not checked for those parameters")
    assert False


if __name__ == "__main__":
  main()
