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

import argparse
import json
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder # pylint: disable=no-name-in-module

def main():
  """Main method to run the example code."""

  parser = argparse.ArgumentParser()
  parser.add_argument("--name", help="the test name")
  parser.add_argument("--cmaddr", help="IP:port for CS system")
  args = parser.parse_args()

  # Get params from compile metadata
  with open(f"{args.name}/out.json", encoding='utf-8') as json_file:
    compile_data = json.load(json_file)

  LOCAL_OUT_SZ = int(compile_data['params']['LOCAL_OUT_SZ'])
  LOCAL_IN_SZ = int(compile_data['params']['LOCAL_IN_SZ'])

  #  if not, must redo the routing
  width = 2
  height = 2

  M = LOCAL_OUT_SZ * height # number of rows of matrix A
  N = LOCAL_IN_SZ * width   # number of cols of matrix A

  print(f"M = {M}, N = {N}, width = {width}, height = {height}")

  # prepare host data and reference solution
  np.random.seed(2)
  A = np.arange(M*N).reshape(M, N).astype(np.float32)
  x = np.arange(N).reshape(N, 1).astype(np.float32) + 100
  b = np.arange(M).reshape(M, 1).astype(np.float32) + 200

  # compute reference
  Ax = np.matmul(A, x)
  r = b - Ax
  nrm_r = np.linalg.norm(r, np.inf)

  print(f"nrm_r = |b - A*x| = {nrm_r}")

  # prepare the simulation
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

  print(f"`nrm_r`     from CPU:\n{nrm_r}")
  print(f"`nrm_r_cs`  from CS:\n{nrm_r_cs}")

  dr = abs(nrm_r - nrm_r_cs[0])
  print(f"|nrm_r - nrm_r_cs| = {dr}")

  assert np.allclose(nrm_r, nrm_r_cs[0], 1.e-5)
  print("\nSUCCESS!")

if __name__ == "__main__":
  main()
