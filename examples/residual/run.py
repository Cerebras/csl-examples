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
from typing import List, Optional
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
  parser.add_argument("--debug", \
        help="show A, x, and b", action="store_true")
  parser.add_argument(
      "--cslc",
      required=False,
      default=CSLC,
      help="The path to the csl compiler. Defaults to '" + CSLC + "'",
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

  args = parser.parse_args()

  return args


def csl_compile(name: str, compile_flag: bool, cslc: str, LOCAL_OUT_SZ: int, LOCAL_IN_SZ: int, \
    width: int, height: int, file_config: str)  -> List[str]:
  """Generate ELFs for the layout, one ELF per PE"""

  # halo has size 1
  fabric_width = width + 2
  fabric_height = height + 2
  csl_cmd = f"{cslc} {file_config} --fabric-dims={fabric_width},{fabric_height} \
  --fabric-offsets=1,1 --params=LOCAL_OUT_SZ:{LOCAL_OUT_SZ},LOCAL_IN_SZ:{LOCAL_IN_SZ} -o {name}"
  print(f"[csl_compile] command line for CSL: {csl_cmd}")
  if compile_flag:
    result = os.system(csl_cmd)
    if result > 0:
      print("ERROR: CSL fails\n")
      exit(1)
  else:
    print(f"MUST CHECK: no -c flag, the user has to compile layout.csl with above command")

  pes = glob(f"{name}/bin/out_[0-9]*.elf")

  return pes


# usage:
# without -c flag: the user has to compile manually
#    cslc layout.csl --fabric-dims=4,4  --fabric-offsets=1,1 \
#        --params=LOCAL_OUT_SZ:3,LOCAL_IN_SZ:2
#    python run.py -m=6 -n=4
#
# with -c flag: the compilation is triggered by run.py
#    python run.py -m=6 -n=4 -c
#
#
# show the content of A,x,b with --debug flag
#    python run.py -m=6 -n=4 --debug
#
# Assumption:
#  m and n must be even so each PE has the same dimension of local tensor A
#
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

  # only support 2-by-2 rectangle
  assert M == (LOCAL_OUT_SZ*height), "M must be even"
  assert N == (LOCAL_IN_SZ*width), "N must be even"

  print(f"M = {M}, N = {N}, width = {width}, height = {height}")

  # prepare host data and reference solution
  np.random.seed(2)
  A = np.random.rand(M, N).astype(np.float32)
  x = np.random.rand(N, 1).astype(np.float32)
  b = np.random.rand(M, 1).astype(np.float32)

  if args.debug:
    print(f"A = {A}")
    print(f"x = {x}")
    print(f"b = {b}")

  Ax = np.matmul(A, x)
  r = b - Ax
  #r = b - (A @ x)

  nrm_r = np.linalg.norm(r, np.inf)

  print(f"nrm_r = |b - A*x| = {nrm_r}")

  # prepare the simulation

  # core dump after execution is complete
  # layout of a rectangle
  code_csl = "layout.csl"

  # compile csl files and generate compilation ELFs
  elf_list = csl_compile(args.name, args.compile, args.cslc, LOCAL_OUT_SZ, LOCAL_IN_SZ, \
    width, height, code_csl)

  # run simulation
  nrm_r_cs = run(args.name, elf_list, width, height, A, M, N, x, b,
                 args.cmaddr)

  print(f"`nrm_r`     from CPU:\n{nrm_r}")
  print(f"`nrm_r_cs`  from CS1 (1-by-1 matrix):\n{nrm_r_cs}")

  dr = abs(nrm_r - nrm_r_cs[(0, 0)])
  print(f"|nrm_r - nrm_r_cs| = {dr}")

  assert np.allclose(nrm_r, nrm_r_cs[(0, 0)], 1.e-5)
  print("\nSUCCESS!")


def run(name: str, elf_list: List[str], width: int, height: int, A: np.ndarray, M: int, N: int, \
    x: np.ndarray, b: np.ndarray, cmaddr: Optional[str]) -> np.ndarray:
  """setup input/output ports
     run simulation
     gather residual norm from the output port """

  LOCAL_OUT_SZ = M // height
  LOCAL_IN_SZ = N // width

  #  i = N*(i/N) + (i % N)  ==> PE_y = (i/N)
  #  b_global(i) --> b_local(index) in PE(-1, PE_y)
  iportmap_b = f"{{ b[i=0:{M-1}][j=0] -> [PE[-1, i//{LOCAL_OUT_SZ}] -> \
        index[i%{LOCAL_OUT_SZ}]] }}"
  print(f"[run] iportmap_b = {iportmap_b}")

  #  x_global(i) --> x_local(index) in PE(PE_x, -1)
  iportmap_x = f"{{ x[i=0:{N-1}][j=0] -> [PE[i//{LOCAL_IN_SZ}, -1] ->  \
        index[i%{LOCAL_IN_SZ}]] }}"
  print(f"[run] iportmap_x = {iportmap_x}")

  # P1.0 sends |r| to the east
  oportmap_nrm_r = f"{{ nrm_r[i=0:0][j=0] -> [PE[{width}, 0] -> index[i]] }}"
  print(f"[run] oportmap_nrm_r = {oportmap_nrm_r}")

  print(f"[run] elf_list = {elf_list}")
  runner = CSELFRunner(elf_list, cmaddr=cmaddr)

  # generate data ELFs for local tensor A
  assert width == 2, "residual_eval only supports 2-by-2 rectangles"
  assert height == 2, "residual_eval only supports 2-by-2 rectangles"

  size_locA = LOCAL_OUT_SZ*LOCAL_IN_SZ
  A_pes = np.zeros((width, height, size_locA), np.float32, order='F')

  for row in range(M):
    # row = py * LOCAL_OUT_SZ + i
    i = row % LOCAL_OUT_SZ
    py = row // LOCAL_OUT_SZ
    for col in range(N):
      # col = px * LOCAL_IN_SZ + j
      j = col % LOCAL_IN_SZ
      px = col // LOCAL_IN_SZ
      # local tensor is column-major order
      A_pes[(px, py, i+j*LOCAL_OUT_SZ)] = A[(row, col)]

  runner.set_symbol_rect("A", A_pes, offset=(1, 1))

  # the following colors are from layout.csl
  # color FABRIC_X = 0 ;
  # color FABRIC_B = 1 ;
  # color TXACT    = 2 ;
  c_x_in = 0
  c_b_in = 1
  c_nrm_r_out = 2
  print(f"[run] input/output must have different colors, c_x_in={c_x_in}, \
        c_b_in={c_b_in}, c_nrm_r_out ={c_nrm_r_out}")
  runner.add_input_tensor(c_b_in, iportmap_b, b, sentinel=None)
  runner.add_input_tensor(c_x_in, iportmap_x, x, sentinel=None)
  runner.add_output_tensor(c_nrm_r_out, oportmap_nrm_r, dtype=np.float32)

  # Run the computation on a simulated wafer or "simfabric".
  # This is a blocking call that stops when all output tensors are received.
  sim_out_path = f"{name}/bin/core.out"
  runner.connect_and_run(sim_out_path)

  nrm_r_cs = runner.out_tensor_dict["nrm_r"]

  return nrm_r_cs


if __name__ == "__main__":
  main()
