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

# pylint: disable=line-too-long

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

   The vector x is distributed into columns from the north of the rectangle.
   The first row receives x from the fabric, then broadcasts x into other rows.
   One can distribute x to the diagonal PEs from the west of the rectangle,
   then broadcasts x to columns.
   This requires two colors, one to receive x and the other to send out x.

   The vector b is distributed into rows of the first column from the west of the rectangle.

   P1.0 computes |b-A*x| and sends it out.

   One can use the following command to check the landing log of P0.0,
    sdk_debug_shell wavelet-trace --artifact_dir . --x 1 --y 1 trace

"""


import os
import argparse
from glob import glob
from typing import List
import subprocess
import numpy as np

from cerebras.elf.cs_elf_runner import CSELFRunner

FILE_PATH = os.path.realpath(__file__)
RESIDUAL_DIR = os.path.dirname(FILE_PATH)
BENCHMARKS_DIR = os.path.dirname(RESIDUAL_DIR)
CSL_DIR = os.path.dirname(BENCHMARKS_DIR)
CSLC = os.path.join(CSL_DIR, "build") + "/bin/cslc"

def parse_args():
  """ parse the command line """

  parser = argparse.ArgumentParser(description="residual parameters.")
  parser.add_argument("-m", type=int,  \
        help="number of rows of the matrix A")
  parser.add_argument("-n", type=int,  \
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
    compile_flag: bool,
    LOCAL_OUT_SZ: int,
    LOCAL_IN_SZ: int
    ) -> List[str]:
  """Generate ELFs for the layout, one ELF per PE"""

  comp_dir = elf_dir

  MEMCPYH2D_DATA_1 = 0
  MEMCPYH2D_DATA_2 = 1
  MEMCPYH2D_DATA_3 = 2

  MEMCPYD2H_DATA_1 = 3

  if compile_flag:
    args = []
    args.append(cslc) # command
    args.append(file_config) # file
    args.append(f"--fabric-dims={fabric_width},{fabric_height}") # options
    args.append("--fabric-offsets=4,1") # options
    args.append(f"--params=width:{width},height:{height}") # options
    if MEMCPYH2D_DATA_1 is not None:
      args.append(f"--params=MEMCPYH2D_DATA_1_ID:{MEMCPYH2D_DATA_1}") # options
    if MEMCPYH2D_DATA_2 is not None:
      args.append(f"--params=MEMCPYH2D_DATA_2_ID:{MEMCPYH2D_DATA_2}") # options
    if MEMCPYH2D_DATA_3 is not None:
      args.append(f"--params=MEMCPYH2D_DATA_3_ID:{MEMCPYH2D_DATA_3}") # options
    if MEMCPYD2H_DATA_1 is not None:
      args.append(f"--params=MEMCPYD2H_DATA_1_ID:{MEMCPYD2H_DATA_1}") # options

    args.append(f"--params=LOCAL_OUT_SZ:{LOCAL_OUT_SZ},LOCAL_IN_SZ:{LOCAL_IN_SZ}") # options

    args.append(f"-o={comp_dir}")
    args.append("--memcpy")

    print(f"subprocess.check_call(args = {args}")
    subprocess.check_call(args)
  else:
    print("[csl_compile_core] use pre-compile ELFs")

  elfs = glob(f"{comp_dir}/bin/out_[0-9]*.elf")

  west_dir = os.path.join(elf_dir, "west")
  elfs_west = glob(f"{west_dir}/bin/out_[0-9]*.elf")

  east_dir = os.path.join(elf_dir, "east")
  elfs_east = glob(f"{east_dir}/bin/out_[0-9]*.elf")

  return elfs + elfs_west + elfs_east


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

  # text file containing the simulator logs
  sim_log = os.path.join(args.name, "sim.log")

  print(f"code_csl = {code_csl}")
  print(f"sim_log = {sim_log}")

  fabric_offset_x = 1
  fabric_offset_y = 1
  fabric_width = 0
  fabric_height = 0
  if args.fabric_dims:
    w_str, h_str = args.fabric_dims.split(",")
    fabric_width = int(w_str)
    fabric_height = int(h_str)

  if fabric_width == 0 or fabric_height == 0:
    fabric_width = fabric_offset_x + width + 5 + 1
    fabric_height = fabric_offset_y + height + 1

  print(f"fabric_width = {fabric_width}, fabric_height = {fabric_height}")

  # compile csl files and generate compilation ELFs
  elf_list = csl_compile(
      args.cslc,
      width,
      height,
      code_csl,
      args.name,
      fabric_width,
      fabric_height,
      args.compile,
      LOCAL_OUT_SZ,
      LOCAL_IN_SZ)

  c_h2d = [0, 1, 2]
  c_d2h = [3]

  simulator = CSELFRunner(elf_list, debug_mode=True, cmaddr=args.cmaddr, \
        height=height, width=width, input_colors=set(c_h2d), output_colors=set(c_d2h))

  # A is M-by-N
  iportmap_A = f"{{ A[j=0:{M-1}][i=0:{N-1}] -> [PE[i//{LOCAL_IN_SZ}, j//{LOCAL_OUT_SZ}] -> \
        index[i%{LOCAL_IN_SZ}, j%{LOCAL_OUT_SZ}]] }}"
  print(f"iportmap_A = {iportmap_A}")

  # x distributes to {py = 0}
  iportmap_x = f"{{ x[i=0:{N-1}][j=0] -> [PE[i//{LOCAL_IN_SZ}, 0] ->  \
        index[i%{LOCAL_IN_SZ}]] }}"
  print(f"iportmap_x = {iportmap_x}")

  # b distributes to {px = 0}
  #  i = N*(i/N) + (i % N)  ==> PE_y = (i/N)
  iportmap_b = f"{{ b[i=0:{M-1}][j=0] -> [PE[0, i//{LOCAL_OUT_SZ}] -> \
        index[i%{LOCAL_OUT_SZ}]] }}"
  print(f"iportmap_b = {iportmap_b}")

  # |b-A*x| is from P1.0
  oportmap_nrm_r = "{ nrm_r[i=0:0][j=0] -> [PE[1, 0] -> index[i]] }"
  print(f"oportmap_nrm_r = {oportmap_nrm_r}")

  simulator.add_input_tensor(c_h2d[0], iportmap_A, A)
  simulator.add_input_tensor(c_h2d[1], iportmap_x, x)
  simulator.add_input_tensor(c_h2d[2], iportmap_b, b)

  simulator.add_output_tensor(c_d2h[0], oportmap_nrm_r, np.float32)

  simulator.connect_and_run()

  if args.cmaddr is None:
    # move simulation log and core dump to the given folder
    mv_sim_cmd = f"mv sim.log {sim_log}"
    os.system(mv_sim_cmd)

    mv_simfab_traces_cmd = f"mv simfab_traces {args.name}"
    ret = os.system(mv_simfab_traces_cmd)
    err_msg = f"{args.name}/simfab_traces exists, please remove it first"
    assert ret == 0, err_msg

  nrm_r_cs = simulator.out_tensor_dict["nrm_r"]

  print(f"`nrm_r`     from CPU:\n{nrm_r}")
  print(f"`nrm_r_cs`  from CS1 (1-by-1 matrix):\n{nrm_r_cs}")

  dr = abs(nrm_r - nrm_r_cs[(0, 0)])
  print(f"|nrm_r - nrm_r_cs| = {dr}")

  assert np.allclose(nrm_r, nrm_r_cs[(0, 0)], 1.e-5)
  print("\nSUCCESS!")


if __name__ == "__main__":
  main()
