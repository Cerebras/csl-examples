#!/usr/bin/env cs_python

import argparse
import json
import struct
import numpy as np

from cerebras.elf.cs_elf_runner import CSELFRunner

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='Name of the benchmark')
parser.add_argument('--cmaddr', help='IP:port for CS system')

args = parser.parse_args()
name = args.name

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

# Path to ELF files
elf_paths = [f"{name}/bin/out_0_0.elf"]

# Set the seed so that CI results are deterministic
np.random.seed(seed=7)

# Generate four 64-bit random integers, and turn them into two 128-bit numbers
num = np.random.randint((1 << 31) - 1, (1 << 63) - 1, size=4, dtype=np.int64)
left = (int(num[0]) << 64) | int(num[1])
right = (int(num[2]) << 64) | int(num[3])

tensors = np.concatenate([intToWords(left), intToWords(right)])

# Parse the compile metadata
compile_data = None
with open(f"{name}/out.json", encoding="utf-8") as json_file:
  compile_data = json.load(json_file)
assert compile_data is not None
compile_colors = compile_data["colors"]
receiveColor = int(compile_colors["recvColor"])
triggerColor = int(compile_colors["triggerColor"])
outputColor = int(compile_colors["outputColor"])

runner = CSELFRunner(elf_paths, cmaddr=args.cmaddr)

runner.add_input_tensor(receiveColor, ("WEST", 0), tensors, sentinel=triggerColor)

output_port_map = "{out_tensor[idx=0:15] -> [PE[-1,0] -> index[idx]]}"
runner.add_output_tensor(outputColor, output_port_map, np.uint16)

# Simulate ELF file and wait for the program to complete
runner.connect_and_run()

# Read the result from the output
result = wordsToInt(runner.out_tensor_dict["out_tensor"])

print("****************")
print(f"{hex(left)} * {hex(right)} is {hex(result)}")
print("****************")

np.testing.assert_equal(result, left * right)
print("SUCCESS!")
