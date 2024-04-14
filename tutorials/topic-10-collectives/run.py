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


import argparse
import json
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder # pylint: disable=no-name-in-module

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='the test name')
parser.add_argument('--cmaddr', help='IP:port for CS system')
args = parser.parse_args()
dirname = args.name

# Parse the compile metadata
with open(f"{dirname}/out.json", encoding="utf-8") as json_file:
  compile_data = json.load(json_file)
params = compile_data["params"]
Pw = int(params["Pw"])
Ph = int(params["Ph"])
chunk_size = int(params["chunk_size"])
print(f"Pw = width of the core = {Pw}")
print(f"Ph = height of the core = {Ph}")
print(f"chunk_size = {chunk_size}")

Nx = Pw*chunk_size
Ny = Ph*chunk_size

print(f"Nx = {Nx}, Ny = {Ny}")

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

sym_broadcast_data = runner.get_id("broadcast_data")
sym_scatter_data = runner.get_id("scatter_data")
sym_broadcast_recv = runner.get_id("broadcast_recv")
sym_faddh_result = runner.get_id("faddh_result")
sym_gather_recv = runner.get_id("gather_recv")

runner.load()
runner.run()

print("step 1: copy mode H2D(broadcast_data) to 1st column PEs")
broadcast_data = np.ones((Ph, 1, Nx)).astype(np.float32)
runner.memcpy_h2d(sym_broadcast_data, broadcast_data.ravel(), 0, 0, 1, Ph, Nx, \
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=True)

print("step 2: copy mode H2D(scatter_data) to 1st row PEs")
scatter_data = np.ones((1, Pw, Ny)).astype(np.int32)
runner.memcpy_h2d(sym_scatter_data, scatter_data.ravel(), 0, 0, Pw, 1, Ny, \
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=True)

print("step 3: call f_run_x to test broadcast and reduction")
runner.launch("f_run_x", nonblock=False)

print("step 4: call f_run_y to test scatter and gather")
runner.launch("f_run_y", nonblock=False)

print("step 5: copy mode D2H(broadcast_recv)")
# broadcast on x: Px=0 broadcasts data to all other PEs
# broadcast_recv(y, x=0) = 0
# broadcast_recv(y, x !=0) = ones
broadcast_recv_1d = np.zeros(Ph*Pw*Nx, np.float32)
runner.memcpy_d2h(broadcast_recv_1d, sym_broadcast_recv, 0, 0, Pw, Ph, Nx, \
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
broadcast_recv = broadcast_recv_1d.reshape((Ph, Pw, Nx))

print("step 6: copy mode D2H(faddh_result) from 1st column PEs")
# reduce(broadcast_recv) to Px=0
faddh_result_1d = np.zeros(Ph*Nx, np.float32)
runner.memcpy_d2h(faddh_result_1d, sym_faddh_result, 0, 0, 1, Ph, Nx, \
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
faddh_result = faddh_result_1d.reshape((Ph, 1, Nx))

print("step 7: copy mode D2H(gather_recv) from 1st row PEs")
gather_recv_1d = np.zeros(Pw*Ny, np.int32)
runner.memcpy_d2h(gather_recv_1d, sym_gather_recv, 0, 0, Pw, 1, Ny, \
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
gather_recv = gather_recv_1d.reshape((1, Pw, Ny))

runner.stop()

# verify broadcast on x-direction
correct_broadcast_recv = np.ones(Nx).astype(np.float32)
for y in range(Ph):
  for x in range(Pw):
    if x == 0:
      continue
    np.testing.assert_equal(broadcast_recv[y, x], correct_broadcast_recv)

# verify faddh_result at 1st column PEs
# reduce on x: reduce(broadcast_recvs) to Px=0
# where broadcast_recvs(y, x=0) = 0
#       broadcast_recvs(y, x != 0) = ones
correct_faddh_result = np.full(Nx, (Pw-1), dtype=np.float32)
for y in range(Ph):
  np.testing.assert_equal(faddh_result[y, 0], correct_faddh_result)

# verify gather_recv at 1st row PEs
correct_gather_recv = np.ones(Ny).astype(np.int32)
for x in range(Pw):
  np.testing.assert_equal(gather_recv[0, x], correct_gather_recv)

print("SUCCESS")
