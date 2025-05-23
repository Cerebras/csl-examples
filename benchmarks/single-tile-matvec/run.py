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

# pylint: disable=line-too-long,too-many-function-args

import argparse
import csv
import json
import math
import struct
import time

import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import (  # pylint: disable=no-name-in-module
    MemcpyDataType, MemcpyOrder, SdkRuntime,
)


def parse_args():
  """parse the command line"""

  parser = argparse.ArgumentParser(description="single tile matvec run parameters")
  parser.add_argument("--name", required=False, default="out", help="prefix of ELF files")
  parser.add_argument("--cmaddr", required=False, default="", help="IP:port for CS system")
  parser.add_argument("--verify", action="store_true", help="Verify Y computation")

  args = parser.parse_args()
  return args


def float_to_hex(f):
  return hex(struct.unpack("<I", struct.pack("<f", f))[0])


def make_u48(words):
  return words[0] + (words[1] << 16) + (words[2] << 32)


def sub_ts(words):
  return make_u48(words[3:]) - make_u48(words[0:3])


def main():
  """Main method to run the example code."""

  args = parse_args()

  name = args.name
  cmaddr = args.cmaddr
  verify = args.verify

  # Parse the compile metadata
  with open(f"{name}/out.json", encoding="utf-8") as json_file:
    compile_data = json.load(json_file)

  nb = int(compile_data["params"]["tile_size"])
  width = int(compile_data["params"]["width"])
  height = int(compile_data["params"]["height"])
  iters = int(compile_data["params"]["iters"])

  print(f"nb = {nb}")
  print(f"width = {width}")
  print(f"height = {height}")
  print(f"iters = {iters}")

  # Calculate alignment and padding to avoid bank conflicts
  align = 16
  multiple = int(align / 4)
  padded_nb = math.ceil(nb / multiple) * multiple

  #############
  # Run
  #############

  start = time.time()

  # Instantiate runner
  runner = SdkRuntime(name, cmaddr=cmaddr)

  # Device symbols for memcpy
  A_symbol = runner.get_id("A")
  x_symbol = runner.get_id("x")
  y_symbol = runner.get_id("y")
  symbol_maxmin_time = runner.get_id("maxmin_time")

  # Load and begin run
  runner.load()
  runner.run()

  # Construct A data and copy random A matrix PE (0,0) for verification
  A_mat = np.random.rand(nb, nb)
  A_data = np.zeros(width * height * (nb * padded_nb + 1), dtype=np.float32)

  for w in range(width):
    for h in range(height):
      for i in range(nb):
        for j in range(nb):
          A_data[(h * width + w) * (nb * padded_nb + 1) + j * padded_nb + i + 1] = A_mat[i, j]

  print()
  print("Beginning run.")
  print("Copy A matrices to device...")
  runner.memcpy_h2d(
      A_symbol,
      A_data,
      0,
      0,
      width,
      height,
      nb * padded_nb + 1,
      streaming=False,
      data_type=MemcpyDataType.MEMCPY_32BIT,
      order=MemcpyOrder.ROW_MAJOR,
      nonblock=False,
  )

  # Construct x data and copy random x vector to PE (0,0) for verification
  x_vec = np.random.rand(nb)
  x_data = np.zeros(width * height * nb, dtype=np.float32)
  for w in range(width):
    for h in range(height):
      x_data[(h * width + w) * nb:(h * width + w) * nb + nb] = x_vec

  print("Copy x vectors to device...")
  runner.memcpy_h2d(
      x_symbol,
      x_data,
      0,
      0,
      width,
      height,
      nb,
      streaming=False,
      data_type=MemcpyDataType.MEMCPY_32BIT,
      order=MemcpyOrder.ROW_MAJOR,
      nonblock=False,
  )

  # Launch the compute kernel
  print("Launch kernel...")
  runner.call("compute", [], nonblock=False)

  # Copy back timestamps from device
  data = np.zeros((width * height * 3, 1), dtype=np.uint32)
  runner.memcpy_d2h(
      data,
      symbol_maxmin_time,
      0,
      0,
      width,
      height,
      3,
      streaming=False,
      data_type=MemcpyDataType.MEMCPY_32BIT,
      order=MemcpyOrder.ROW_MAJOR,
      nonblock=False,
  )
  maxmin_time_hwl = data.view(np.float32).reshape((height, width, 3))
  print("Copied back timestamps.")

  # Copy back data array from device
  if verify:
    data = np.zeros((width * height * padded_nb, 1), dtype=np.uint32)
    runner.memcpy_d2h(
        data,
        y_symbol,
        0,
        0,
        width,
        height,
        padded_nb,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )
    y_device_array = data.view(np.float32).reshape((height, width, padded_nb))
    print("Copied back Y array.")

  print("Done.")

  # End walltime timer
  runner.stop()
  end = time.time()
  walltime = end - start

  ###########
  # Verify
  ###########

  if verify:
    print("Test y result is as expected on each PE...")
    expected = A_mat @ x_vec
    for w in range(width):
      for h in range(height):
        np.testing.assert_allclose(y_device_array[h, w, :nb], expected, atol=0.0001, rtol=0)
    print("SUCCESS!")

  #################################
  # Calculate mem accesses and FLOP
  #################################

  # STANDARD read/writes
  # Read full x, read each column of V stack, write full y = nb + nb*nb + nb
  # = nb*nb + 2*nb
  #
  # 4 bytes per elem. Mem = 4 * (nb*nb + 2*nb)
  #                       = 4*nb*nb + 8*nb

  # ACTUAL read/writes
  # Read full x; read each col of V stack; read, write full y nb times
  # = nb + nb*nb + 2*nb*nb
  # = 3*nb*nb + nb
  #
  # 4 bytes per elem. Mem = 4 * (3*nb*nb + nb)
  #                       = 12*nb*nb + 4*nb

  # Floating point operations
  # Compute A_ij * x_j for each i, j = nb * nb
  # For each row of A, reduction uses nb - 1 adds = nb * (nb-1)
  # = nb * nb + nb * (nb-1)
  # = 2*nb*nb - nb

  total_relative_accesses = width * height * (4 * nb * nb + 8 * nb)
  total_absolute_accesses = width * height * (12 * nb * nb + 4 * nb)
  total_flop = width * height * (2 * nb * nb - nb)

  #######################
  # Calculate cycle count
  #######################

  tsc_tensor_d2h = np.zeros(6).astype(np.uint16)
  min_cycles = math.inf
  max_cycles = 0

  for w in range(width):
    for h in range(height):
      hex_t0 = int(float_to_hex(maxmin_time_hwl[(h, w, 0)]), base=16)
      hex_t1 = int(float_to_hex(maxmin_time_hwl[(h, w, 1)]), base=16)
      hex_t2 = int(float_to_hex(maxmin_time_hwl[(h, w, 2)]), base=16)
      tsc_tensor_d2h[0] = hex_t0 & 0x0000FFFF
      tsc_tensor_d2h[1] = (hex_t0 >> 16) & 0x0000FFFF
      tsc_tensor_d2h[2] = hex_t1 & 0x0000FFFF
      tsc_tensor_d2h[3] = (hex_t1 >> 16) & 0x0000FFFF
      tsc_tensor_d2h[4] = hex_t2 & 0x0000FFFF
      tsc_tensor_d2h[5] = (hex_t2 >> 16) & 0x0000FFFF

      cycles = sub_ts(tsc_tensor_d2h)
      if cycles < min_cycles:
        min_cycles = cycles
        min_w = w
        min_h = h
      if cycles > max_cycles:
        max_cycles = cycles
        max_w = w
        max_h = h

  #####################
  # Calculate bandwidth
  #####################

  # Calculate in bytes/sec and FLOP/sec for program rectangle
  secs = max_cycles / 850000000.0
  relative_bw = total_relative_accesses / secs * iters
  absolute_bw = total_absolute_accesses / secs * iters
  flops_sec = total_flop / secs

  # Convert to Petabytes/sec and PetaFLOPS
  relative_bw /= 1.0e15
  absolute_bw /= 1.0e15
  flops_sec /= 1.0e15

  # Scale to program rectangle
  scale_factor = (994.0 * 750.0) / (width * height)
  scale_relative_bw = relative_bw * scale_factor
  scale_absolute_bw = absolute_bw * scale_factor
  scale_flops_sec = flops_sec * scale_factor

  #################
  # Generate output
  #################

  print()
  print(f"Real walltime: {walltime}s")
  print()
  print("Cycle Counts:")
  print("Min cycles (", min_w, ", ", min_h, "): ", min_cycles)
  print("Max cycles (", max_w, ", ", max_h, "): ", max_cycles)
  print()
  print("Accesses and FLOP Information:")
  print("Relative accesses (bytes): ", total_relative_accesses)
  print("Absolute accesses (bytes): ", total_absolute_accesses)
  print("FP operations:             ", total_flop)
  print()
  print("Bandwidth and FLOPS Information:")
  print("Relative BW (PB/s): ", relative_bw)
  print("Absolute BW (PB/s): ", absolute_bw)
  print("PetaFLOPS:          ", flops_sec)
  print()
  print("Scaled (", width, ",", height, ") to (750,994)...")
  print("Scaled relative BW (PB/s): ", scale_relative_bw)
  print("Scaled absolute BW (PB/s): ", scale_absolute_bw)
  print("Scaled PetaFLOPS:          ", scale_flops_sec)

  # Write a CSV
  csv_name = name + ".csv"
  with open(csv_name, encoding="utf-8", mode="a") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        cmaddr,
        width,
        height,
        iters,
        nb,
        padded_nb,
        min_cycles,
        max_cycles,
        total_relative_accesses,
        total_absolute_accesses,
        relative_bw,
        absolute_bw,
        scale_relative_bw,
        scale_absolute_bw,
        total_flop,
        flops_sec,
        scale_flops_sec,
        walltime,
    ])


if __name__ == "__main__":
  main()
