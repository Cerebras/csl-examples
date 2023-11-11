#!/usr/bin/env cs_python

# Copyright 2023 Cerebras Systems.
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

""" test bandwidth between host and device

    The host connects the device via 100Gbps ethernets. The data is distributed
  from/to couple of I/O channels. The maximum bandwidth of a single channel is
  around 7Gbps (Giga bit per second). In addition, the overhead of TCP is about
  200 us, a non-negligible cost when the transaction is small.

  The bandwidth is affected by the following factors:
  (1) number of I/O channels
      The number of I/O channels is controlled by the flag --channels=<int>
      The more channels, the higher bandwidth
  (2) buffers to hold input/output data to hide the long latency of I/O
      Although The I/O channelsand the core are independent, if the core has a
      heavy computation such that it cannot respond to the I/O request, there is
      a backpressure from the core upstream to the I/O channels. The backpressure
      stalls the data transfer and the host can no longer push the data.
      I/O channel will resume only when the core responds the request,however
      there is a long latency before the core can receive the data.
      To overlap the computaton and communication (or to avoid this long latency)
      , we can insert buffers to hold the data from the I/O channels while the
      core is busy for something else.
      The user can use flag --width-west-buf=<int> to set a buffer for the input
      and the flag --width-east-buf to set a buffer for the output.
      Each PE in the buffer has 46KB fifo to store the data, if a H2D/D2H has
      "pe_length" u32 per PE and "width" PEs per row, it needs
      (pe_length*width)*4/46K columns
  (3) blocking (sync) or nonblocking (async)
      The long latency of I/O can be amortized if multiple requests are combined
      together into one TCP transfer (200 us overhead per TCP transaction). The
      runtime can aggregate multiple nonblocking H2D/D2H commands implicitly.
      The user can set paramerer 'nonblock=True' to enable async operations.

  The framework of bandwidthTest is
  ---
       sync   // synchronize all PEs to sample the reference clock
       tic()  // record start time
       for j = 0:loop_count
          H2D or D2H (sync or async)
       end
       toc()  // record end time
  ---

  To record the elapsed time on host may not show the true timing because the
  runtime may not start the transaction when the user calls a H2D/D2H command.
  For example, the runtime can aggregate multiple nonblocking commands together.
  Instead, this bandwidhTest samples the timing on the device.

  The strategy is to record "start" time and "end" time of H2D/D2H on each PE and
  to compute the elapsed time by the different of max/min of these two numbers.
  However the tsc timer is not synchronized and could differ a lot if we take max
  or min operation on the timer. To obtain the reliable timing info, we need to
  synchronize all PEs and use one PE to trigger the timer such that all PEs can
  start at "the same" time. The "sync" operation can sample the reference clock
  which is the initial time t0 for all PEs.
  Even we shift the "start clock" by the "reference clock", each PE does not have
  the same number because of the propagation delay of the signal. The delay of
  "start clock" is about the dimension of the WSE.

  Here is the list of parameters:
    The flag --loop_count=<int> decides how many H2Ds/D2Hs are called.
    The flag --d2h measures the bandwidth of D2H, otherwise bandwidth of H2D is
        measured.
    The flag --channels specifies the number of I/O channels, no bigger than 16.

  The tic() samples "time_start" and toc() samples "time_end". The sync() samples
  "time_ref" which is used to shift "time_start" and "time_end".
  The elapsed time is measured by
       cycles_send = max(time_end) - min(time_start)

  The overall runtime is computed via the following formula
       time_send = (cycles_send / 0.85) *1.e-3 us
  where a PE runs with clock speed 850MHz

  The bandwidth is calculated by
       bandwidth = ((wvlts * 4)/time_send)*loop_count
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

from bw_cmd_parser import parse_args



def float_to_hex(f):
  return hex(struct.unpack('<I', struct.pack('<f', f))[0])

def make_u48(words):
  return words[0] + (words[1] << 16) + (words[2] << 32)

def cast_uint32(x):
  if isinstance(x, (np.float16, np.int16, np.uint16)):
    z = x.view(np.uint16)
    val = np.uint32(z)
  elif isinstance(x, (np.float32, np.int32, np.uint32)):
    val = x.view(np.uint32)
  elif isinstance(x, int):
    val = np.uint32(x)
  elif isinstance(x, float):
    z = np.float32(x)
    val = z.view(np.uint32)
  else:
    raise RuntimeError(f"type of x {type(x)} is not supported")

  return val

def csl_compile_core(
    cslc: str,
    width: int,  # width of the core
    height: int, # height of the core
    pe_length: int,
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
    args.append(f"--params=width:{width},height:{height},pe_length:{pe_length}")
    args.append(f"--params=LAUNCH_ID:{LAUNCH}")
    args.append(f"--params=C0_ID:{C0}")
    args.append(f"--params=C1_ID:{C1}")
    args.append(f"--params=C2_ID:{C2}")
    args.append(f"--params=C3_ID:{C3}")
    args.append(f"--params=C4_ID:{C4}")

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


def hwl_2_oned_colmajor(
    height: int,
    width: int,
    pe_length: int,
    A_hwl: np.ndarray
):
  """
    Given a 3-D tensor A[height][width][pe_length], transform it to
    1D array by column-major
  """
  A_1d = np.zeros(height*width*pe_length, np.float32)
  idx = 0
  for l in range(pe_length):
    for w in range(width):
      for h in range(height):
        A_1d[idx] = A_hwl[(h, w, l)]
        idx = idx + 1
  return A_1d


# How to compile:
#  <path/to/cslc> bw_sync_layout.csl --fabric-dims=12,7 --fabric-offsets=4,1 \
#    --params=width:5,height:5,pe_length:5 \
#    --params=LAUNCH_ID:5 --params=C0_ID:0 --params=C1_ID:1 --params=C2_ID:2 \
#    --params=C3_ID:3 --params=C4_ID:4 \
#    -o=latest --memcpy --channels=1 --width-west-buf=0 --width-east-buf=0
# or
#  python run.py -m=5 -n=5 -k=5 --latestlink latest --channels=1 \
#    --width-west-buf=0 --width-east-buf=0 \
#    --compile-only --driver=<path/to/cslc>
#
# How to run:
#  python run.py -m=5 -n=5 -k=5 --latestlink latest --channels=1 \
#   --width-west-buf=0 --width-east-buf=0 \
#   --run-only --loop_count=1
#
# To run a WSE, add --cmaddr=<IP address of WSE>
#
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
  loop_count = args.loop_count

  print(f"width = {width}, height = {height}, pe_length={pe_length}, loop_count = {loop_count}")

  np.random.seed(2)
  # A is h-by-w-by-l
  A = np.arange(height*width*pe_length).reshape(height, width, pe_length).astype(np.float32)

  A_1d = hwl_2_oned_colmajor(height, width, pe_length, A)

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

  # core dump after execution is complete
  #core_path = os.path.join(dirname, "core.out")

  # text file containing the simulator logs
  sim_log = os.path.join(dirname, "sim.log")

  # layout of a rectangle
  code_csl = "bw_sync_layout.csl"

  C0 = 0
  C1 = 1
  C2 = 2
  C3 = 3
  C4 = 4
  LAUNCH = 5

  csl_compile_core(
      cslc,
      width,
      height,
      pe_length,
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
      channels,
      width_west_buf,
      width_east_buf
  )
  if args.compile_only:
    print("COMPILE ONLY: EXIT")
    return

  # output tensor via D2H
  E_1d = np.zeros(height*width*pe_length, np.float32)

  memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
  simulator = SdkRuntime(dirname, cmaddr=args.cmaddr)

  symbol_A = simulator.get_id("A")
  symbol_time_memcpy = simulator.get_id("time_memcpy")
  symbol_time_ref = simulator.get_id("time_ref")

  simulator.load()
  simulator.run()

  print("step 1: sync() synchronizes all PEs and records reference clock")
  simulator.call("f_sync", [], nonblock=True)

  print("step 2: tic() records time_start")
  simulator.call("f_tic", [], nonblock=True)

  if args.d2h:
    for j in range(loop_count):
      print(f"step 3: measure D2H with loop_count = {loop_count}, {j}-th")
      simulator.memcpy_d2h(E_1d, symbol_A, 0, 0, width, height, pe_length,\
          streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)
  else:
    for j in range(loop_count):
      print(f"step 3: measure H2D with loop_count = {loop_count}, {j}-th")
      simulator.memcpy_h2d(symbol_A, A_1d, 0, 0, width, height, pe_length,\
          streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=True)

  print("step 4: toc() records time_end")
  simulator.call("f_toc", [], nonblock=False)

  print("step 5: prepare (time_start, time_end)")
  simulator.call("f_memcpy_timestamps", [], nonblock=False)

  print("step 6: D2H (time_start, time_end)")
  # time_start/time_end is of type u16[3]
  # {time_start, time_end} is packed into three f32
  time_memcpy_1d_f32 = np.zeros(height*width*3, np.float32)
  simulator.memcpy_d2h(time_memcpy_1d_f32, symbol_time_memcpy, 0, 0, width, height, 3,\
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
  time_memcpy_hwl = np.reshape(time_memcpy_1d_f32, (height, width, 3), order='C')

  print("step 7: prepare reference clock")
  simulator.call("f_reference_timestamps", [], nonblock=False)

  print("step 8: D2H reference clock")
  # time_ref is of type u16[3], packed into two f32
  time_ref_1d_f32 = np.zeros(height*width*2, np.float32)
  simulator.memcpy_d2h(time_ref_1d_f32, symbol_time_ref, 0, 0, width, height, 2,\
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
  time_ref_hwl = np.reshape(time_ref_1d_f32, (height, width, 2), order='C')

  #simulator.stop(core_path)
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

  # time_start = start time of H2D/D2H
  time_start = np.zeros((height, width)).astype(int)
  # time_end = end time of H2D/D2H
  time_end = np.zeros((height, width)).astype(int)
  word = np.zeros(3).astype(np.uint16)
  for w in range(width):
    for h in range(height):
      hex_t0 = int(float_to_hex(time_memcpy_hwl[(h, w, 0)]), base=16)
      hex_t1 = int(float_to_hex(time_memcpy_hwl[(h, w, 1)]), base=16)
      hex_t2 = int(float_to_hex(time_memcpy_hwl[(h, w, 2)]), base=16)
      word[0] = hex_t0 & 0x0000ffff
      word[1] = (hex_t0 >> 16) & 0x0000ffff
      word[2] = hex_t1 & 0x0000ffff
      time_start[(h, w)] = make_u48(word)
      word[0] = (hex_t1 >> 16) & 0x0000ffff
      word[1] = hex_t2 & 0x0000ffff
      word[2] = (hex_t2 >> 16) & 0x0000ffff
      time_end[(h, w)] = make_u48(word)

  # time_ref = reference clock
  time_ref = np.zeros((height, width)).astype(int)
  word = np.zeros(3).astype(np.uint16)
  for w in range(width):
    for h in range(height):
      hex_t0 = int(float_to_hex(time_ref_hwl[(h, w, 0)]), base=16)
      hex_t1 = int(float_to_hex(time_ref_hwl[(h, w, 1)]), base=16)
      word[0] = hex_t0 & 0x0000ffff
      word[1] = (hex_t0 >> 16) & 0x0000ffff
      word[2] = hex_t1 & 0x0000ffff
      time_ref[(h, w)] = make_u48(word)
  # adjust the reference clock by the propagation delay
  for py in range(height):
    for px in range(width):
      time_ref[(py, px)] = time_ref[(py, px)] - (px + py)

  # shift time_start and time_end by time_ref
  time_start = time_start - time_ref
  time_end = time_end - time_ref

  # cycles_send = time_end[(h,w)] - time_start[(h,w)]
  # 850MHz --> 1 cycle = (1/0.85) ns = (1/0.85)*1.e-3 us
  # time_send = (cycles_send / 0.85) *1.e-3 us
  # bandwidth = (((wvlts-1) * 4)/time_send) MBS
  wvlts = height*width*pe_length
  min_time_start = time_start.min()
  max_time_end = time_end.max()
  cycles_send = max_time_end - min_time_start
  time_send = (cycles_send / 0.85) *1.e-3
  bandwidth = ((wvlts * 4)/time_send)*loop_count
  print(f"wvlts = {wvlts}, loop_count = {loop_count}")
  print(f"cycles_send = {cycles_send} cycles")
  print(f"time_send = {time_send} us")
  print(f"bandwidth = {bandwidth} MB/S ")


if __name__ == "__main__":
  main()
