#!/usr/bin/env cs_python

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

# pylint: disable=too-many-function-args
""" Test row or column broadcast
    The kernel is the same as bandwidthTest.
    The bandwidth calculation follows bandwidthTest.

    Here is the list of parameters:
    -m=<int> specifies the height of the core.
    -n=<int> specifies the width of the core.
    -k=<int> specifies the maximum number of elements per PE in the core.
    --roi_px=<int> specifies the starting column index of region of interest
    --roi_py=<int> specifies the starting row index of region of interest
    --roi_w=<int> specifies the width of region of interest
    --roi_h=<int> specifies the height of region of interest
    --channels specifies the number of I/O channels, no bigger than 16.
"""

import random
import struct

import numpy as np
from cmd_parser import parse_args

from cerebras.sdk.runtime.sdkruntimepybind import (  # pylint: disable=no-name-in-module
    MemcpyDataType, MemcpyOrder, SdkRuntime,
)


def float_to_hex(f):
  return hex(struct.unpack("<I", struct.pack("<f", f))[0])


def make_u48(words):
  return words[0] + (words[1] << 16) + (words[2] << 32)


def main():
  """Main method to run the example code."""

  random.seed(127)

  args, dirname = parse_args()

  height = args.m
  width = args.n
  pe_length = args.k
  use_col_major = args.use_col_major
  is_row_bcast = args.is_row_bcast
  loop_count = args.loop_count

  print(f"core: width = {width}, height = {height}, pe_length={pe_length}")

  np.random.seed(2)
  if is_row_bcast:
    print("row broadcast mode: only prepare data for 1 column")
    # A is h-by-1-by-l
    A = (np.arange(height * 1 * pe_length).reshape(height, 1, pe_length).astype(np.uint32))
  else:
    print("column broadcast mode: only prepare data for 1 row")
    # A is 1-by-w-by-l
    A = (np.arange(1 * width * pe_length).reshape(1, width, pe_length).astype(np.uint32))
  print(f"shape(A) = {A.shape}")
  print(f"A = {A}")

  px = args.roi_px
  py = args.roi_py
  pw = args.roi_w
  ph = args.roi_h

  print(f"ROI: px = {px}, py = {py}, pw = {pw}, ph = {ph}")

  assert px >= 0, "px must be non-negative"
  assert py >= 0, "px must be non-negative"
  assert pw <= width, "pw must not be greater than width"
  assert ph <= height, "ph must not be greater than height"

  # extract ROI from A
  if is_row_bcast:
    B = A[py:(py + ph), 0:, 0:]
  else:
    B = A[0:, px:(px + pw), 0:]
  print(f"shape(B) = {B.shape}")
  print(f"B = {B}")

  bx, by, bz = B.shape
  if is_row_bcast:
    assert bx == ph
    assert by == 1
    assert bz == pe_length
  else:
    assert bx == 1
    assert by == pw
    assert bz == pe_length

  print(f"use_col_major = {use_col_major}")
  if use_col_major:
    B_1d = B.T.ravel()
  else:
    B_1d = B.ravel()

  print("store ELFs and log files in the folder ", dirname)

  memcpy_dtype = MemcpyDataType.MEMCPY_32BIT

  runner = SdkRuntime(
      dirname,
      suppress_simfab_trace=True,
      # msg_level="DEBUG",
      cmaddr=args.cmaddr,
  )

  symbol_A = runner.get_id("A")
  symbol_time_memcpy = runner.get_id("time_memcpy")
  symbol_time_ref = runner.get_id("time_ref")

  runner.load()
  runner.run()

  print("step 1: sync() synchronizes all PEs and records reference clock")
  runner.call("f_sync", [], nonblock=True)

  print("step 2: tic() records time_start")
  runner.call("f_tic", [], nonblock=True)

  print(f"len(B_1d) = {len(B_1d)}")
  print(f"B_1d = {B_1d}")
  for _ in range(loop_count):
    if is_row_bcast:
      print("step 1: memcpy_h2d_rowbcast(B)")
      runner.memcpy_h2d_rowbcast(
          symbol_A,
          B_1d,
          px,
          py,
          pw,
          ph,
          pe_length,
          streaming=False,
          data_type=memcpy_dtype,
          order=(MemcpyOrder.COL_MAJOR if use_col_major else MemcpyOrder.ROW_MAJOR),
          nonblock=True,
      )
    else:
      print("step 1: memcpy_h2d_colbcast(B)")
      runner.memcpy_h2d_colbcast(
          symbol_A,
          B_1d,
          px,
          py,
          pw,
          ph,
          pe_length,
          streaming=False,
          data_type=memcpy_dtype,
          order=(MemcpyOrder.COL_MAJOR if use_col_major else MemcpyOrder.ROW_MAJOR),
          nonblock=True,
      )

  print("step 4: toc() records time_end")
  runner.call("f_toc", [], nonblock=False)

  print("step 5: prepare (time_start, time_end)")
  runner.call("f_memcpy_timestamps", [], nonblock=False)

  print("step 6: D2H (time_start, time_end)")
  # time_start/time_end is of type u16[3]
  # {time_start, time_end} is packed into three f32
  time_memcpy_1d_f32 = np.zeros(height * width * 3, np.float32)
  runner.memcpy_d2h(
      time_memcpy_1d_f32,
      symbol_time_memcpy,
      0,
      0,
      width,
      height,
      3,
      streaming=False,
      data_type=memcpy_dtype,
      order=MemcpyOrder.ROW_MAJOR,
      nonblock=False,
  )
  time_memcpy_hwl = np.reshape(time_memcpy_1d_f32, (height, width, 3), order="C")

  print("step 7: prepare reference clock")
  runner.call("f_reference_timestamps", [], nonblock=False)

  print("step 8: D2H reference clock")
  # time_ref is of type u16[3], packed into two f32
  time_ref_1d_f32 = np.zeros(height * width * 2, np.float32)
  runner.memcpy_d2h(
      time_ref_1d_f32,
      symbol_time_ref,
      0,
      0,
      width,
      height,
      2,
      streaming=False,
      data_type=memcpy_dtype,
      order=MemcpyOrder.ROW_MAJOR,
      nonblock=False,
  )
  time_ref_hwl = np.reshape(time_ref_1d_f32, (height, width, 2), order="C")

  print("step 9: D2H(A)")
  E_1d = np.zeros(height * width * pe_length, A.dtype)
  runner.memcpy_d2h(
      E_1d,
      symbol_A,
      0,
      0,
      width,
      height,
      pe_length,
      streaming=False,
      data_type=memcpy_dtype,
      order=MemcpyOrder.COL_MAJOR,
      nonblock=False,
  )

  runner.stop()

  print("DONE")

  # E is h-by-w-by-l
  E_hwl = np.reshape(E_1d, (height, width, pe_length), order="F")
  print(f"E_hwl (from device) = {E_hwl}")

  # B_ext is the expected result
  B_ext = (np.zeros(height * width * pe_length).reshape(height, width, pe_length).astype(A.dtype))
  if is_row_bcast:
    # copy B to each column of ROI
    for w in range(pw):
      B_ext[py:(py + ph), (px + w):(px + w + 1), 0:] = B
  else:
    # copy B to each row of ROI
    for h in range(ph):
      B_ext[(py + h):(py + h + 1), px:(px + pw), 0:] = B
  print(f"B_ext = {B_ext}")

  print("check E_hwl == B_ext")
  assert np.allclose(E_hwl.ravel(), B_ext.ravel(), 0)

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
      word[0] = hex_t0 & 0x0000FFFF
      word[1] = (hex_t0 >> 16) & 0x0000FFFF
      word[2] = hex_t1 & 0x0000FFFF
      time_start[(h, w)] = make_u48(word)
      word[0] = (hex_t1 >> 16) & 0x0000FFFF
      word[1] = hex_t2 & 0x0000FFFF
      word[2] = (hex_t2 >> 16) & 0x0000FFFF
      time_end[(h, w)] = make_u48(word)

  # time_ref = reference clock
  time_ref = np.zeros((height, width)).astype(int)
  word = np.zeros(3).astype(np.uint16)
  for w in range(width):
    for h in range(height):
      hex_t0 = int(float_to_hex(time_ref_hwl[(h, w, 0)]), base=16)
      hex_t1 = int(float_to_hex(time_ref_hwl[(h, w, 1)]), base=16)
      word[0] = hex_t0 & 0x0000FFFF
      word[1] = (hex_t0 >> 16) & 0x0000FFFF
      word[2] = hex_t1 & 0x0000FFFF
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
  wvlts = pw * ph * pe_length
  min_time_start = time_start.min()
  max_time_end = time_end.max()
  cycles_send = max_time_end - min_time_start
  time_send = (cycles_send / 0.85) * 1.0e-3
  bandwidth = ((wvlts * 4) / time_send) * loop_count
  print(f"ROI: pw = {pw}, ph= {ph}, pe_length={pe_length}")
  print(f"wvlts = {wvlts}, loop_count = {loop_count}")
  print(f"cycles_send = {cycles_send} cycles")
  print(f"time_send = {time_send} us")
  print(f"bandwidth = {bandwidth} MB/S ")


if __name__ == "__main__":
  main()
