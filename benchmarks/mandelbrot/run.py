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


from itertools import product
import argparse
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder # pylint: disable=no-name-in-module

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="the test name")
parser.add_argument("--cmaddr", help="IP:port for CS system")
args = parser.parse_args()
dirname = args.name

rows = 16
cols = 16

MEMCPYD2H_DATA_1 = 1
memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

runner.load()
runner.run()

runner.launch("f_mandelbrot", nonblock=False)

# The following ISL maps describes the output tensor based on CSELFRunner
#   oport = f"{{ R[i=0:{rows - 1}, j=0:{cols - 1}, k=0:2] -> [PE[4, i // 4] -> index[i,j,k]] }}"
# The last column is P4,y
#
# R.shpe = (16, 16, 3)
#   P4.0 --> index[i = 0,1,2,3, j=0:15, k=0:2]
#   P4.1 --> index[i = 4,5,6,7, j=0:15, k=0:2]
#   P4.2 --> index[i = 8,9,10,11, j=0:15, k=0:2]
#   P4.3 --> index[i = 12,13,14,15, j=0:15, k=0:2]
#
# The last column of PEs (4 PEs) receives the data
# Each PE receives (rows/4)*cols*3
packets = np.zeros((rows*cols*3), np.float32)
# The packets has the dimension (h,w,l) = (4,1,4*16*3) in row-major order
# we can reshape it by (16,16,3)

# ROI = last column of PEs
#     = P3.y where y = 0,1,2,3
#     = (px=3,px=0) with (w=1,h=4)
runner.memcpy_d2h(packets, MEMCPYD2H_DATA_1, 3, 0, 1, 4, 4*cols*3, \
    streaming=True, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
R = packets.reshape((rows, cols, 3))

runner.stop()

iters = R[:, :, 2]
# A simple in-terminal representation of the Mandelbrot set
print(iters)

ref = np.zeros((rows, cols))

x_lo, x_hi = -2.0, 1.0
y_lo, y_hi = -1.5, 1.5
max_iters = 32

for r, c in product(range(rows), range(cols)):
  x = c * (x_hi - x_lo) / (cols - 1) + x_lo
  y = r * (y_hi - y_lo) / (rows - 1) + y_lo

  val = np.csingle(x + y * 1j)

  for i in range(max_iters + 1):
    if abs(val) >= 2.0:
      break
    val = val * val + (x + y * 1j)
  ref[r, c] = i

np.testing.assert_equal(ref, iters)

print("SUCCESS")
