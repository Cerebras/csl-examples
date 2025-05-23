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


import argparse
import json
import struct
import numpy as np

from cerebras.sdk.sdk_utils import memcpy_view, input_array_to_u32
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder # pylint: disable=no-name-in-module

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='Name of the benchmark')
parser.add_argument('--cmaddr', help='IP:port for CS system')

args = parser.parse_args()
dirname = args.name

numBits = 256
byteCount = numBits // 8
wordCount = numBits // 16

def intToWords(int_value):
  byteList = int_value.to_bytes(byteCount, 'little')
  groupedList = [byteList[i:i+2] for i in range(0, len(byteList), 2)]
  wordList = [struct.unpack("<H", group)[0] for group in groupedList]
  return np.array(wordList, dtype=np.uint16)

def wordsToInt(words):
  byteList = b''.join(int(word).to_bytes(2, 'little') for word in words)
  return int.from_bytes(byteList, 'little')

# Set the seed so that CI results are deterministic
np.random.seed(seed=7)

# Generate four 64-bit random integers, and turn them into two 128-bit numbers
num = np.random.randint((1 << 31) - 1, (1 << 63) - 1, size=4, dtype=np.int64)
left = (int(num[0]) << 64) | int(num[1])
right = (int(num[2]) << 64) | int(num[3])

tensors = np.concatenate([intToWords(left), intToWords(right)])

# Parse the compile metadata
with open(f"{dirname}/out.json", encoding="utf-8") as json_file:
  compile_data = json.load(json_file)

params = compile_data["params"]
MEMCPYH2D_DATA_1 = int(params["MEMCPYH2D_DATA_1_ID"])
MEMCPYD2H_DATA_1 = int(params["MEMCPYD2H_DATA_1_ID"])
print(f"MEMCPYH2D_DATA_1 = {MEMCPYH2D_DATA_1}")
print(f"MEMCPYD2H_DATA_1 = {MEMCPYD2H_DATA_1}")

memcpy_dtype = MemcpyDataType.MEMCPY_16BIT
runner = SdkRuntime(dirname, cmaddr=args.cmaddr)

runner.load()
runner.run()

print("call f_run to setup the number of H2D wavelets")
runner.launch("f_run", np.int16(tensors.size), nonblock=False)

print("streaming H2D: the kernel computes and sends out the result when H2D is done")
input_tensor_u32 = input_array_to_u32(tensors, 1, tensors.size)
runner.memcpy_h2d(MEMCPYH2D_DATA_1, input_tensor_u32, 0, 0, 1, 1, input_tensor_u32.size, \
    streaming=True, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=True)

print("streaming D2H")
# The D2H buffer must be of type u32
output_tensor_u32 = np.zeros(16, np.uint32)
runner.memcpy_d2h(output_tensor_u32, MEMCPYD2H_DATA_1, 0, 0, 1, 1, 16, \
    streaming=True, data_type=memcpy_dtype, order=MemcpyOrder.COL_MAJOR, nonblock=False)
# remove upper 16-bit of each u32
out_tensor = memcpy_view(output_tensor_u32, tensors.dtype)

runner.stop()

# Read the result from the output
result = wordsToInt(out_tensor)

print("****************")
print(f"{hex(left)} * {hex(right)} is {hex(result)}")
print("****************")

np.testing.assert_equal(result, left * right)
print("SUCCESS!")
