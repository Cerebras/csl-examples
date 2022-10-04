#!/usr/bin/env cs_python

import argparse
import json
from glob import glob
import numpy as np

from cerebras.elf.cs_elf_runner import CSELFRunner

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='the test name')
parser.add_argument('--cmaddr', help='IP:port for CS system')
args = parser.parse_args()
name = args.name

# Path to ELF files
elf_paths = glob(f"{name}/bin/out_*.elf")

# Simulate ELF files
runner = CSELFRunner(elf_paths, cmaddr=args.cmaddr)

A = np.array([[42.0, 42.0, 42.0, 42.0, 42.0],
              [42.0, 42.0, 42.0, 42.0, 42.0],
              [42.0, 42.0, 42.0, 42.0, 42.0],
              [42.0, 42.0, 42.0, 42.0, 42.0],
              [42.0, 42.0, 42.0, 42.0, 42.0]]).astype(np.float16)
B = np.array([10, 20, 30, 40, 50]).astype(np.int16)

def transformation(value: np.array, coeff1: float, coeff2: float, weight: np.array):
  return np.multiply(value, coeff1 + weight) + np.multiply(value, coeff2 + weight)

def reduction(array):
  return sum(array)

np.random.seed(seed=7)

# Parse the compile metadata
compile_data = None
with open(f"{name}/out.json", 'rt', encoding='utf-8') as json_file:
  compile_data = json.load(json_file)
assert compile_data is not None
compile_params = compile_data['params']
size = int(compile_params['size'])

weights = np.random.random(size).astype(np.float16)
input_port_map = f"{{weights[idx=0:{size - 1}] -> [PE[-1,0] -> index[idx]]}}"
inColor = 2 # The input color from which the kernel will read data.
runner.add_input_tensor(inColor, input_port_map, weights)

output_port_map = f"{{sqrt_result[idx=0:{size - 1}] -> [PE[1,0] -> index[idx]]}}"
output_color = 0 # The output color to which the kernel will send data.
runner.add_output_tensor(output_color, output_port_map, np.float16)

runner.connect_and_run()

# Square-root example
sqrt_result = runner.out_tensor_dict["sqrt_result"]
expected = np.sqrt(np.diag(A))
np.testing.assert_equal(sqrt_result, expected)

# Transformation example
expected = transformation(np.diag(A), 2.0, 6.0, weights)
np.fill_diagonal(A, expected)
actual = runner.get_symbol(1, 1, "A", np.float16)
np.testing.assert_equal(actual.reshape((5, 5)), A)

# Reduction example
sum_result = np.array([reduction(B)], dtype=np.int16)
expected = runner.get_symbol(1, 1, "sum", np.int16)
np.testing.assert_equal(sum_result, expected)

print("SUCCESS!")
