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
compile_params = compile_data["params"]
HIST_W = int(compile_params["HIST_WIDTH"])
HIST_H = int(compile_params["HIST_HEIGHT"])
INPUT_SIZE = int(compile_params["INPUT_SIZE"])
NUM_BUCKETS = int(compile_params["NUM_BUCKETS"])
BUCKET_SIZE = int(compile_params["BUCKET_SIZE"])

# Set the seed so that CI results are deterministic
np.random.seed(seed=7)

MEMCPYH2D_DATA_1 = int(compile_params["MEMCPYH2D_DATA_1_ID"])
MEMCPYD2H_DATA_1 = int(compile_params["MEMCPYD2H_DATA_1_ID"])
print(f"MEMCPYH2D_DATA_1 = {MEMCPYH2D_DATA_1}")
print(f"MEMCPYD2H_DATA_1 = {MEMCPYD2H_DATA_1}")

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

sym_inputs = runner.get_id("inputs")
sym_buckets = runner.get_id("buckets")

runner.load()
runner.run()

# Randomly generate the input values.
inputs = np.random.randint(
    0,
    HIST_W*HIST_H*NUM_BUCKETS*BUCKET_SIZE,
    (HIST_W, HIST_H, INPUT_SIZE)
).astype(np.uint32)

# inputs: w-h-l with order w=0, h=1, l=2
# inputs_hwl is h-w-l, so the permutation order is 1,0,2
inputs_hwl = np.transpose(inputs, [1, 0, 2])
# inputs_1d is 1D layout of inputs_hwl in col-major order
inputs_1d = inputs_hwl.T.ravel()

print("step 1: copy mode H2D: prepare inputs for all PEs")
runner.memcpy_h2d(sym_inputs, inputs_1d, 0, 0, HIST_W, HIST_H, INPUT_SIZE,\
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=True)

print("step 2: call f_run to setup runtime coordinates and send out the data")
runner.launch("f_run", nonblock=False)

# calculate CPU reference
# "o" is h-w-l
o = np.zeros((HIST_H, HIST_W, NUM_BUCKETS), dtype=np.uint32)
for i in range(HIST_W):
  for j in range(HIST_H):
    for n in range(INPUT_SIZE):
      v = inputs[i][j][n]
      v2 = v // BUCKET_SIZE
      bucket = v2 % NUM_BUCKETS
      v3 = v2 // NUM_BUCKETS
      row = v3 % HIST_W
      col = v3 // HIST_W
      o[col][row][bucket] += 1

print("step 3: streaming mode D2H: receive tally's signal at P(x=w-1,y=0)")
result_tensor = np.zeros(1, np.uint32)
runner.memcpy_d2h(result_tensor, MEMCPYD2H_DATA_1, HIST_W-1, 0, 1, 1, 1, \
    streaming=True, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)
print(f"result_tensor: {result_tensor}")
np.testing.assert_equal(result_tensor, HIST_W * HIST_H * INPUT_SIZE)

print("step 4: copy mode D2H: receive buckets")
# buckets_1d is 1D layout of h-w-l in col-major order
buckets_1d = np.zeros(HIST_H*HIST_W*NUM_BUCKETS, np.uint32)
runner.memcpy_d2h(buckets_1d, sym_buckets, 0, 0, HIST_W, HIST_H, NUM_BUCKETS,\
    streaming=False, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)
ans = buckets_1d.reshape((HIST_H, HIST_W, NUM_BUCKETS), order='F')

runner.stop()

print(f"\ninput for a {HIST_H}x{HIST_W} fabric with {INPUT_SIZE} inputs/PE")
# "inputs" is w-h-l while inputs_hwl is h-w-l
print(inputs_hwl)
print("\nsimfab buckets")
# "ans" is h-w-l
print(ans)
print("\ncpu buckets")
# "o" is h-w-l
print(o)

print("\ndiff:")
print(ans - o)

assert (ans == o).all()
print("SUCCESS!")
