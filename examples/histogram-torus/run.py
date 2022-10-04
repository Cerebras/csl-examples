#!/usr/bin/env cs_python
# pylint: disable=line-too-long

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

# Parse the compile metadata
compile_data = None
with open(f"{name}/out.json", encoding="utf-8") as json_file:
  compile_data = json.load(json_file)
assert compile_data is not None
compile_colors = compile_data["colors"]
compile_params = compile_data["params"]
HIST_W = int(compile_params["HIST_WIDTH"])
HIST_H = int(compile_params["HIST_HEIGHT"])
INPUT_SIZE = int(compile_params["INPUT_SIZE"])
NUM_BUCKETS = int(compile_params["NUM_BUCKETS"])
BUCKET_SIZE = int(compile_params["BUCKET_SIZE"])
OUT_COLOR = int(compile_colors["OUT_COLOR"])

# Set the seed so that CI results are deterministic
np.random.seed(seed=7)

elfs = glob(f"{name}/bin/out_[0-9]*.elf")
runner = CSELFRunner(elfs, cmaddr=args.cmaddr)

# Randomly generate the input values.
inputs = np.random.randint(
    0,
    HIST_W*HIST_H*NUM_BUCKETS*BUCKET_SIZE,
    (HIST_W, HIST_H, INPUT_SIZE)
).astype(np.uint32)

# create ELFs to load values onto fabric
runner.set_symbol_rect("inputs", inputs, offset=(1, 1))
cols = np.full((HIST_H, HIST_W), np.arange(HIST_W), np.uint32).T
runner.set_symbol_rect("PE_X", cols, offset=(1, 1))
rows = np.full((HIST_W, HIST_H), np.arange(HIST_H), np.uint32)
runner.set_symbol_rect("PE_Y", rows, offset=(1, 1))

# calculate CPU reference
o = np.zeros((HIST_W, HIST_H, NUM_BUCKETS), dtype=np.uint32)
for i in range(HIST_W):
  for j in range(HIST_H):
    for n in range(INPUT_SIZE):
      v = inputs[i][j][n]
      v2 = v // BUCKET_SIZE
      bucket = v2 % NUM_BUCKETS
      v3 = v2 // NUM_BUCKETS
      row = v3 % HIST_W
      col = v3 // HIST_W
      o[row][col][bucket] += 1

out_port_map = f"{{out_tensor[idx=0:0] -> [PE[{HIST_W-1},-1] -> index[idx]]}}"
runner.add_output_tensor(OUT_COLOR, out_port_map, np.uint32)

runner.connect_and_run()

# check that tally output was as expected
result_tensor = runner.out_tensor_dict["out_tensor"]
print(f"result_tensor: {result_tensor}")
np.testing.assert_equal(result_tensor, HIST_W * HIST_H * INPUT_SIZE)

# get buckets values on all PEs
rect = ((1, 1), (HIST_W, HIST_H))
ans = runner.get_symbol_rect(rect, "buckets", np.uint32)

print(f"\ninput for a {HIST_W}x{HIST_H} fabric with {INPUT_SIZE} inputs/PE")
print(inputs.T)
print("\nsimfab buckets")
print(ans.T)
print("\ncpu buckets")
print(o.T)

print("\ndiff:")
print(ans.T - o.T)

assert (ans == o).all()
print("SUCCESS!")
