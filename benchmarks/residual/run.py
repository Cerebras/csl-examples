#!/usr/bin/env cs_python

# Copyright 2024 Cerebras Systems.
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


""" Compute |b-A*x| using a 2-by-2 PE rectangle

   The 2-by-2 rectangle is surrounded by a halo of size 1.
   The halo is used to route the input and output data between the host and the device.
   It does not impact the layout index of the kernel code.
   For example, the kernel has 2-by-2 PEs, with the index P0.0, P1.0, P0.1, P1.1
   in the layout/routing configuration.
   The compiler generates ELFs out_0_0.elf, out_0_1.elf, out_1_0.elf and out_1_1.elf.
   However the user needs global coordinate (including halo) for debugging, for example
   P0.0 of the kernel is P1.1 when the user calls sdk_debug_shell to dump the trace or
   landing log.

   Each PE computes A*x and does a row reduction, so last column has the result b - A*x.
   Then last column computes |b-A*x| via a column reduction.

   To simplify the example, the dimensions M and N are assumed even.
   Three functions gemv, axpy and nrminf are used to compute y=A*x, y=y+alpha*x and
   |x|_inf locally.
   Such functions are imported as modules via gemv.csl, axpy.csl and nrminf.csl.
   The arrays A, x and y are passed into the function as pointers.

   The vector x is distributed into columns. The first row receives x from the fabric,
   then broadcasts x into other rows.

   The vector b is distributed into rows of the first column.

   P1.0 computes |b-A*x| which is sent out..

   One can use the following command to check the landing log of P0.0,
    sdk_debug_shell wavelet-trace --artifact_dir . --x 1 --y 1 trace

"""


import os
import argparse
from pathlib import Path
from typing import Optional
import shutil
import subprocess
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder # pylint: disable=no-name-in-module

FILE_PATH = os.path.realpath(__file__)
RESIDUAL_DIR = os.path.dirname(FILE_PATH)
BENCHMARKS_DIR = os.path.dirname(RESIDUAL_DIR)
CSL_DIR = os.path.dirname(BENCHMARKS_DIR)
CSLC = os.path.join(CSL_DIR, "build") + "/bin/cslc"


def cast_uint32(x):
  if isinstance(x, (np.float16, np.int16, np.uint16)):
    z = x.view(np.uint16)
    return np.uint32(z)
  if isinstance(x, (np.float32, np.int32, np.uint32)):
    return x.view(np.uint32)
  if isinstance(x, int):
    return np.uint32(x)

  raise RuntimeError(f"type of x {type(x)} is not supported")



def parse_args():
  """ parse the command line """

  parser = argparse.ArgumentParser(description="residual parameters.")
  parser.add_argument("-m", type=int,
                      help="number of rows of the matrix A")
  parser.add_argument("-n", type=int,
                      help="number of columns of the matrix A. If A is square, \
                            n is the dimension of the matrix and m is not used")
  parser.add_argument(
      "--cslc",
      required=False,
      default=CSLC,
      help=f"The path to the csl compiler. Defaults to '{CSLC}'",
  )
  parser.add_argument(
      "-c", "--compile", action="store_true", help="Compile the code."
  )
  parser.add_argument(
      "--name",
      required=False,
      default="out",
      help="prefix of ELF files",
  )
  parser.add_argument("--cmaddr", help="IP:port for CS system")
  parser.add_argument(
      "--fabric-dims",
      help="Fabric dimension, i.e. <W>,<H>")

  parser.add_argument(
      "--width-west-buf",
      default=0, type=int,
      help="width of west buffer")
  parser.add_argument(
      "--width-east-buf",
      default=0, type=int,
      help="width of east buffer")
  parser.add_argument(
      "--n_channels",
      default=1, type=int,
      help="Number of memcpy \"channels\" (LVDS/streamers for both input and output)  to use \
            when memcpy support is compiled with this program. If this argument is not present, \
            or is 0, then the previous single-LVDS version is compiled.")
  parser.add_argument(
      "--arch",
      help="wse1 or wse2. Default is wse2 when not supplied.")

  args = parser.parse_args()

  return args


def csl_compile(
    cslc: str,
    width: int,
    height: int,
    file_config: str,
    elf_dir: str,
    fabric_width: int,
    fabric_height: int,
    core_fabric_offset_x: int,
    core_fabric_offset_y: int,
    compile_flag: bool,
    arch: Optional[str],
    LOCAL_OUT_SZ: int,
    LOCAL_IN_SZ: int,
    n_channels: int,
    width_west_buf: int,
    width_east_buf: int
    ):
  """Generate ELFs for the layout, one ELF per PE"""

  comp_dir = elf_dir

  if compile_flag:
    args = []
    args.append(cslc) # command
    args.append(file_config) # file
    args.append(f"--fabric-dims={fabric_width},{fabric_height}") # options
    args.append(f"--fabric-offsets={core_fabric_offset_x},{core_fabric_offset_y}") # options
    args.append(f"--params=width:{width},height:{height}") # options
    args.append(f"--params=LOCAL_OUT_SZ:{LOCAL_OUT_SZ},LOCAL_IN_SZ:{LOCAL_IN_SZ}") # options

    args.append(f"-o={comp_dir}")
    if arch is not None:
      args.append(f"--arch={arch}")
    args.append("--memcpy")
    args.append(f"--channels={n_channels}")
    args.append(f"--width-west-buf={width_west_buf}")
    args.append(f"--width-east-buf={width_east_buf}")
    print(f"subprocess.check_call(args = {args}")
    subprocess.check_call(args)
  else:
    print("[csl_compile] use pre-compile ELFs")



def main():
  """Main method to run the example code."""

  args = parse_args()

#  if not, must redo the routing
  width = 2
  height = 2

  if args.m is not None:
    M = args.m
  else:
    M = 6

  if args.n is not None:
    N = args.n
  else:
    N = 4

  LOCAL_OUT_SZ = M // height
  LOCAL_IN_SZ = N // width

  assert M == (LOCAL_OUT_SZ*height), "M must be multiple of LOCAL_OUT_SZ"
  assert N == (LOCAL_IN_SZ*width), "N must be multiple of LOCAL_IN_SZ"

  print(f"M = {M}, N = {N}, width = {width}, height = {height}")

  # prepare host data and reference solution
  np.random.seed(2)
  A = np.arange(M*N).reshape(M, N).astype(np.float32)
  x = np.arange(N).reshape(N, 1).astype(np.float32) + 100
  b = np.arange(M).reshape(M, 1).astype(np.float32) + 200

  Ax = np.matmul(A, x)
  r = b - Ax

  nrm_r = np.linalg.norm(r, np.inf)

  print(f"nrm_r = |b - A*x| = {nrm_r}")

  # prepare the simulation

  # core dump after execution is complete
  # layout of a rectangle
  code_csl = "layout_memcpy.csl"

  n_channels = args.n_channels
  width_west_buf = args.width_west_buf
  width_east_buf = args.width_east_buf
  print(f"n_channels = {n_channels}")
  print(f"width_west_buf = {width_west_buf}, width_east_buf = {width_east_buf}")

  fabric_offset_x = 1
  fabric_offset_y = 1
  fabric_width = 0
  fabric_height = 0
  if args.fabric_dims:
    w_str, h_str = args.fabric_dims.split(",")
    fabric_width = int(w_str)
    fabric_height = int(h_str)

  if fabric_width == 0 or fabric_height == 0:
    fabric_width = fabric_offset_x + 3 + width + 2 + 1 + width_west_buf + width_east_buf
    fabric_height = fabric_offset_y + height + 1

  core_fabric_offset_x = fabric_offset_x + 3 + width_west_buf
  core_fabric_offset_y = fabric_offset_y

  print(f"fabric_width = {fabric_width}, fabric_height = {fabric_height}")
  print(f"core_fabric_offset_x = {core_fabric_offset_x}, ")
  print(f"core_fabric_offset_y = {core_fabric_offset_y}")

  # compile csl files and generate compilation ELFs
  csl_compile(
      args.cslc,
      width,
      height,
      code_csl,
      args.name,
      fabric_width,
      fabric_height,
      core_fabric_offset_x,
      core_fabric_offset_y,
      args.compile,
      args.arch,
      LOCAL_OUT_SZ,
      LOCAL_IN_SZ,
      n_channels,
      width_west_buf,
      width_east_buf)
  if args.compile:
    print("COMPILE ONLY: EXIT")
    return

  memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
  memcpy_order = MemcpyOrder.ROW_MAJOR
  runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

  sym_A = runner.get_id("A")
  sym_x = runner.get_id("x")
  sym_y = runner.get_id("y")
  sym_nrm = runner.get_id("nrm")

  runner.load()
  runner.run()

  # How to transform a 2-D tensor into a cliff distribution with
  # column-major local tensor
  #
  # Example: w=2, h=2, A is 4-by-4 (lh-by-lw)
  # A = |  0  1  2  3 |
  #     |  4  5  6  7 |
  #     |  8  9 10 11 |
  #     | 12 13 14 15 |
  # A1 = A.reshape(2,2,2,2) of the form (h,lh,w,lw)
  # A1 = | | 0  1|  | 4  5| |
  #      | | 2  3|, | 6  7| |
  #      |                  |
  #      | | 8  9|  |12 13| |
  #      | |10 11|, |14 15| |
  # A2 = A1.transpose(0, 2, 3, 1) of the form (h, w, lw, lh)
  # so the local tensor lh-by-lw is col-major
  # A2 = | | 0  4|  | 2  6| |
  #      | | 1  5|, | 3  7| |
  #      |                  |
  #      | | 8 12|  |10 14| |
  #      | | 9 13|, |11 15| |
  # A3 = A2.reshape(2,2,4)
  # A3 = |  0  4  1  5 |
  #      |  2  6  3  7 |
  #      |  8 12  9 13 |
  #      | 10 14 11 15 |
  # A3 is h-w-l

  # |b-A*x| is from P1.0
  oportmap_nrm_r = "{ nrm_r[i=0:0][j=0] -> [PE[1, 0] -> index[i]] }"
  print(f"oportmap_nrm_r = {oportmap_nrm_r}")

  # prepare A, x and b via memcpy
  A1 = A.reshape(height, LOCAL_OUT_SZ, width, LOCAL_IN_SZ)
  A2 = A1.transpose(0, 2, 3, 1)
  A3 = A2.reshape(height, width, LOCAL_OUT_SZ*LOCAL_IN_SZ)
  runner.memcpy_h2d(sym_A, A3.ravel(), 0, 0, width, height, LOCAL_OUT_SZ*LOCAL_IN_SZ,
                    streaming=False, data_type=memcpy_dtype,
                    order=memcpy_order, nonblock=False)

  # x distributes to {py = 0}
  runner.memcpy_h2d(sym_x, x.ravel(), 0, 0, width, 1, LOCAL_IN_SZ,
                    streaming=False, data_type=memcpy_dtype,
                    order=memcpy_order, nonblock=False)

  # b distributes to {px = 0}
  runner.memcpy_h2d(sym_y, b.ravel(), 0, 0, 1, height, LOCAL_OUT_SZ,
                    streaming=False, data_type=memcpy_dtype,
                    order=memcpy_order, nonblock=False)

  # trigger the computation
  runner.launch("bcast_x", nonblock=False)

  # receive |b-A*x| from P1.0
  nrm_r_cs = np.zeros(1, np.float32)
  runner.memcpy_d2h(nrm_r_cs, sym_nrm, 1, 0, 1, 1, 1,
                    streaming=False, data_type=memcpy_dtype,
                    order=memcpy_order, nonblock=False)

  runner.stop()

  if args.cmaddr is None:
    # move simulation log and core dump to the given folder
    dst_log = Path(f"{args.name}/sim.log")
    src_log = Path("sim.log")
    if src_log.exists():
      shutil.move(src_log, dst_log)

    dst_trace = Path(f"{args.name}/simfab_traces")
    src_trace = Path("simfab_traces")
    if dst_trace.exists():
      shutil.rmtree(dst_trace)
    if src_trace.exists():
      shutil.move(src_trace, dst_trace)

  print(f"`nrm_r`     from CPU:\n{nrm_r}")
  print(f"`nrm_r_cs`  from CS1:\n{nrm_r_cs}")

  dr = abs(nrm_r - nrm_r_cs[0])
  print(f"|nrm_r - nrm_r_cs| = {dr}")

  assert np.allclose(nrm_r, nrm_r_cs[0], 1.e-5)
  print("\nSUCCESS!")


if __name__ == "__main__":
  main()
