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

# pylint: disable=too-many-function-args

""" test 7-point stencil

    The Laplacian operator L on 3-dimensional domain can be represented by 7-point
  stencil based on the standard 2nd order Finite Difference Method. The operator form
  with Dirichlet boundary conditions can be written by
         L[u](i,j,k) = u(i+1, j,  k  ) + u(i-1, j,   k  ) +
                       u(i,   j+1,k  ) + u(i,   j-1, k  ) +
                       u(i,   j,  k+1) + u(i,   j,   k-1) +
                      -6*u(i, j, k)
  In general the coefficients of those 7 points can vary. To minimize the memory
  consumption, this example assumes the coefficients are independent of index k and
  whole vector u(i,j,:) is placed in one PE (px=j, py=i).
  The above formula can be re-written by
     c_west   * x[i-1][j  ][k  ] + c_east  * x[i+1][j  ][k  ] +
     c_south  * x[i  ][j-1][k  ] + c_north * x[i  ][j+1][k  ] +
     c_bot    * x[i  ][j  ][k-1] + c_top   * x[i  ][j  ][k+1] +
     c_center * x[i][j][k]
  Each PE only holds 7 coefficients organized by c_west, c_east, c_south, c_north,
  c_bot, c_top and c_center.

  This example provides two modules, one is allreduce and the other is stencil_3d_7pts.
  "allreduce" module can synchronize all PEs to form a reference clock.
  "stencil_3d_7pts" module can compute y = A*x where A is the matrix from 7-point stencil

  The framework is
  ---
       sync()      // synchronize all PEs to sample the reference clock
       tic()       // record start time
       spmv(zdim)  // compute y = A*x
       toc()       // record end time
  ---

  The tic() samples "time_start" and toc() samples "time_end". The sync() samples
  "time_ref" which is used to shift "time_start" and "time_end".
  The elapsed time is measured by
       cycles_send = max(time_end) - min(time_start)

  The overall runtime is computed via the following formula
       time_send = (cycles_send / 0.85) *1.e-3 us
  where a PE runs with clock speed 850MHz

  Each PE needs to gather six f32 from six neighbors, the cost of the communication is
        6*h*w*zDim*4 bytes
  where w-by-h is the core rectangle and zDim is the length of local vector.

  Here is the list of parameters:
    -m=<int> is the height of the core
    -n=<int> is the width of the core
    -k=<int> is size of x and y allocated in the core
    --zDim=<int> is the number of f32 per PE, computed by y = A*x
                 zDim must be not greater than k
    --channels=<int> specifies the number of I/O channels, no bigger than 16
"""


import struct
import os
from typing import Optional
from pathlib import Path
import shutil
import subprocess
import random

import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder # pylint: disable=no-name-in-module

from cmd_parser import parse_args


from util import (
    hwl_2_oned_colmajor,
    oned_to_hwl_colmajor,
    laplacian,
)


def float_to_hex(f):
  return hex(struct.unpack('<I', struct.pack('<f', f))[0])

def make_u48(words):
  return words[0] + (words[1] << 16) + (words[2] << 32)


def csl_compile_core(
    cslc: str,
    width: int,  # width of the core
    height: int, # height of the core
    pe_length: int,
    blockSize: int,
    file_config: str,
    elf_dir: str,
    fabric_width: int,
    fabric_height: int,
    core_fabric_offset_x: int, # fabric-offsets of the core
    core_fabric_offset_y: int,
    use_precompile: bool,
    arch: Optional[str],
    LAUNCH: int,
    C0: int,
    C1: int,
    C2: int,
    C3: int,
    C4: int,
    C5: int,
    C6: int,
    C7: int,
    C8: int,
    channels: int,
    width_west_buf: int,
    width_east_buf: int
):
  if not use_precompile:
    args = []
    args.append(cslc) # command
    args.append(file_config)
    args.append(f"--fabric-dims={fabric_width},{fabric_height}")
    args.append(f"--fabric-offsets={core_fabric_offset_x},{core_fabric_offset_y}")
    args.append(f"--params=width:{width},height:{height},MAX_ZDIM:{pe_length}")
    args.append(f"--params=BLOCK_SIZE:{blockSize}")
    args.append(f"--params=LAUNCH_ID:{LAUNCH}")
    args.append(f"--params=C0_ID:{C0}")
    args.append(f"--params=C1_ID:{C1}")
    args.append(f"--params=C2_ID:{C2}")
    args.append(f"--params=C3_ID:{C3}")
    args.append(f"--params=C4_ID:{C4}")
    args.append(f"--params=C5_ID:{C5}")
    args.append(f"--params=C6_ID:{C6}")
    args.append(f"--params=C7_ID:{C7}")
    args.append(f"--params=C8_ID:{C8}")

    args.append(f"-o={elf_dir}")
    if arch is not None:
      args.append(f"--arch={arch}")
    args.append("--memcpy")
    args.append(f"--channels={channels}")
    args.append(f"--width-west-buf={width_west_buf}")
    args.append(f"--width-east-buf={width_east_buf}")

    print(f"subprocess.check_call(args = {args}")
    subprocess.check_call(args)
  else:
    print("\tuse pre-compile ELFs")



def main():
  """Main method to run the example code."""

  random.seed(127)

  args, dirname = parse_args()

  cslc = "cslc"
  if args.driver is not None:
    cslc = args.driver

  print(f"cslc = {cslc}")

  width_west_buf = args.width_west_buf
  width_east_buf = args.width_east_buf
  channels = args.channels
  assert channels <= 16, "only support up to 16 I/O channels"
  assert channels >= 1, "number of I/O channels must be at least 1"

  print(f"width_west_buf = {width_west_buf}")
  print(f"width_east_buf = {width_east_buf}")
  print(f"channels = {channels}")

  height = args.m
  width = args.n
  pe_length = args.k
  zDim = args.zDim
  blockSize = args.blockSize

  print(f"width = {width}, height = {height}, pe_length={pe_length}, zDim={zDim}, blockSize={blockSize}")
  assert pe_length >= 2, "the maximum size of z must be greater than 1"
  assert zDim <= pe_length, "[0, zDim) cannot exceed the storage"

  np.random.seed(2)
  # A is h-by-w-by-l
  x = np.arange(height*width*pe_length).reshape(height, width, pe_length).astype(np.float32) + 100

  x_1d = hwl_2_oned_colmajor(height, width, pe_length, x, np.float32)

  # stencil coefficients has the following order
  # {c_west, c_east, c_south, c_north, c_bottom, c_top, c_center}
  stencil_coeff = np.zeros((height, width, 7), dtype = np.float32)
  for i in range(height):
    for j in range(width):
      stencil_coeff[(i, j, 0)] = -1 # west
      stencil_coeff[(i, j, 1)] = -2 # east
      stencil_coeff[(i, j, 2)] = -3 # south
      stencil_coeff[(i, j, 3)] = -4 # north
      stencil_coeff[(i, j, 4)] = -5 # bottom
      stencil_coeff[(i, j, 5)] = -6 # top
      stencil_coeff[(i, j, 6)] = 6  # center

  stencil_coeff_1d = hwl_2_oned_colmajor(height, width, 7, stencil_coeff, np.float32)

  y_ref = np.zeros((height, width, pe_length), dtype=np.float32)

  laplacian(stencil_coeff, zDim, x, y_ref)

  # fabric-offsets = 1,1
  fabric_offset_x = 1
  fabric_offset_y = 1
  # starting point of the core rectangle = (core_fabric_offset_x, core_fabric_offset_y)
  # memcpy framework requires 3 columns at the west of the core rectangle
  # memcpy framework requires 2 columns at the east of the core rectangle
  core_fabric_offset_x = fabric_offset_x + 3 + width_west_buf
  core_fabric_offset_y = fabric_offset_y
  # (min_fabric_width, min_fabric_height) is the minimal dimension to run the app
  min_fabric_width = (core_fabric_offset_x + width + 2 + 1 + width_east_buf)
  min_fabric_height = (core_fabric_offset_y + height + 1)

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

  # prepare the simulation
  print('store ELFs and log files in the folder ', dirname)

  # text file containing the simulator logs
  sim_log = os.path.join(dirname, "sim.log")

  # layout of a rectangle
  code_csl = "layout.csl"

  C0 = 0
  C1 = 1
  C2 = 2
  C3 = 3
  C4 = 4
  C5 = 5
  C6 = 6
  C7 = 7
  C8 = 8
  LAUNCH = 10

  csl_compile_core(
      cslc,
      width,
      height,
      pe_length,
      blockSize,
      code_csl,
      dirname,
      fabric_width,
      fabric_height,
      core_fabric_offset_x,
      core_fabric_offset_y,
      args.run_only,
      args.arch,
      LAUNCH,
      C0,
      C1,
      C2,
      C3,
      C4,
      C5,
      C6,
      C7,
      C8,
      channels,
      width_west_buf,
      width_east_buf
  )
  if args.compile_only:
    print("COMPILE ONLY: EXIT")
    return

  memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
  simulator = SdkRuntime(dirname, cmaddr=args.cmaddr)

  symbol_x = simulator.get_id("x")
  symbol_y = simulator.get_id("y")
  symbol_stencil_coeff = simulator.get_id("stencil_coeff")
  symbol_time_buf_u16 = simulator.get_id("time_buf_u16")
  symbol_time_ref = simulator.get_id("time_ref")

  simulator.load()
  simulator.run()

  print(f"copy vector x of type f32")
  # the size of x per PE is pe_length
  simulator.memcpy_h2d(symbol_x, x_1d, 0, 0, width, height, pe_length,\
          streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=True)

  print(f"copy coefficients of type f32")
  # each PE holds 7 coefficients
  simulator.memcpy_h2d(symbol_stencil_coeff, stencil_coeff_1d, 0, 0, width, height, 7,\
          streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=True)

  print("step 1: sync all PEs")
  simulator.launch("f_sync", np.int16(1), nonblock=False)

  print("step 2: tic() records time_start")
  simulator.launch("f_tic", nonblock=True)

  print(f"step 3: compute y = A*x with zDim = {zDim}")
  # positive zDim can be smaller than pe_length
  simulator.launch("f_spmv", np.int16(zDim), nonblock=False)

  print("step 4: toc() records time_end")
  simulator.launch("f_toc", nonblock=False)

  print("step 5: prepare (time_start, time_end)")
  simulator.launch("f_memcpy_timestamps", nonblock=False)

  print("step 6: D2H (time_start, time_end)")
  time_memcpy_hwl_1d = np.zeros(height*width*6, np.uint32)
  simulator.memcpy_d2h(time_memcpy_hwl_1d, symbol_time_buf_u16, 0, 0, width, height, 6,\
    streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT, order=MemcpyOrder.COL_MAJOR, nonblock=False)
  time_memcpy_hwl = oned_to_hwl_colmajor(height, width, 6, time_memcpy_hwl_1d, np.uint16)

  print("step 7: D2H y of type f32")
  y_1d = np.zeros(height*width*pe_length, np.float32)
  simulator.memcpy_d2h(y_1d, symbol_y, 0, 0, width, height, pe_length,\
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)
  y_wse = np.reshape(y_1d, (height, width, pe_length), order='F')

  print("step 8: prepare reference clock")
  simulator.launch("f_reference_timestamps", nonblock=False)

  print("step 9: D2H reference clock")
  time_ref_1d = np.zeros(height*width*3, np.uint32)
  simulator.memcpy_d2h(time_ref_1d, symbol_time_ref, 0, 0, width, height, 3,\
    streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT, order=MemcpyOrder.COL_MAJOR, nonblock=False)
  time_ref_hwl = oned_to_hwl_colmajor(height, width, 3, time_ref_1d, np.uint16)

  simulator.stop()

  if args.cmaddr is None:
    # move simulation log and core dump to the given folder
    dst_log = Path(f"{dirname}/sim.log")
    src_log = Path("sim.log")
    if src_log.exists():
      shutil.move(src_log, dst_log)

    dst_trace = Path(f"{dirname}/simfab_traces")
    src_trace = Path("simfab_traces")
    if dst_trace.exists():
      shutil.rmtree(dst_trace)
    if src_trace.exists():
      shutil.move(src_trace, dst_trace)

  # time_start = start time of spmv
  time_start = np.zeros((height, width)).astype(int)
  # time_end = end time of spmv
  time_end = np.zeros((height, width)).astype(int)
  word = np.zeros(3).astype(np.uint16)
  for w in range(width):
    for h in range(height):
      word[0] = time_memcpy_hwl[(h, w, 0)]
      word[1] = time_memcpy_hwl[(h, w, 1)]
      word[2] = time_memcpy_hwl[(h, w, 2)]
      time_start[(h,w)] = make_u48(word)
      word[0] = time_memcpy_hwl[(h, w, 3)]
      word[1] = time_memcpy_hwl[(h, w, 4)]
      word[2] = time_memcpy_hwl[(h, w, 5)]
      time_end[(h,w)] = make_u48(word)

  # time_ref = reference clock
  time_ref = np.zeros((height, width)).astype(int)
  word = np.zeros(3).astype(np.uint16)
  for w in range(width):
    for h in range(height):
      word[0] = time_ref_hwl[(h, w, 0)]
      word[1] = time_ref_hwl[(h, w, 1)]
      word[2] = time_ref_hwl[(h, w, 2)]
      time_ref[(h, w)] = make_u48(word)

  # adjust the reference clock by the propagation delay
  # the right-bottom PE signals other PEs, the propagation delay is
  #     (h-1) - py + (w-1) - px
  for py in range(height):
    for px in range(width):
      time_ref[(py, px)] = time_ref[(py, px)] - ((width+height-2)-(px + py))

  # shift time_start and time_end by time_ref
  time_start = time_start - time_ref
  time_end = time_end - time_ref

  # cycles_send = time_end[(h,w)] - time_start[(h,w)]
  # 850MHz --> 1 cycle = (1/0.85) ns = (1/0.85)*1.e-3 us
  # time_send = (cycles_send / 0.85) *1.e-3 us
  #
  # each PE needs to gather six f32 from six neighbors, the cost of the communication is
  #      6*h*w*zDim*4 bytes
  #
  # bandwidth = (((wvlts-1) * 4)/time_send) MBS
  wvlts = 6*height*width*zDim
  min_time_start = time_start.min()
  max_time_end = time_end.max()
  cycles_send = max_time_end - min_time_start
  time_send = (cycles_send / 0.85) *1.e-3
  bandwidth = ((wvlts * 4)/time_send)
  print(f"cycles_send = {cycles_send} cycles")
  print(f"time_send = {time_send} us")
  print(f"bandwidth = {bandwidth} MB/S ")

  z = y_ref.ravel() - y_wse.ravel()
  nrm_z = np.linalg.norm(z, np.inf)
  print(f"|y_ref - y_wes| = {nrm_z}")
  np.testing.assert_allclose(y_ref.ravel(), y_wse.ravel(), 1.e-5)
  print("\nSUCCESS!")

if __name__ == "__main__":
  main()
