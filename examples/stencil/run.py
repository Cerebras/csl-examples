#!/usr/bin/env cs_python
# pylint: disable=line-too-long

import argparse
import json
from glob import glob
import numpy as np

from cerebras.elf.cs_elf_runner import CSELFRunner

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='Name of the benchmark')
parser.add_argument('--cmaddr', help='IP:port for CS system')

args = parser.parse_args()
name = args.name


def computeOracle(problemDepth, problemHeight, problemWidth, boundarySize,
                  timeSteps):
  dataWidth = problemWidth + 2 * boundarySize
  dataHeight = problemHeight + 2 * boundarySize
  dataDepth = problemDepth + 2 * boundarySize
  data = np.zeros((2, dataDepth, dataHeight, dataWidth), dtype=np.float16)

  startWIdx = boundarySize
  endWIdx = startWIdx + problemWidth

  startHIdx = boundarySize
  endHIdx = startHIdx + problemHeight

  startDIdx = boundarySize
  endDIdx = startDIdx + problemDepth

  data[0, startDIdx:endDIdx, startHIdx:endHIdx, startWIdx:endWIdx] = 1.0

  for idx in range(timeSteps):
    toggle = idx % 2

    for i in range(problemDepth):
      baseI = boundarySize + i

      for j in range(problemHeight):
        baseJ = boundarySize + j

        for k in range(problemWidth):
          sumVar = 0.0
          baseK = boundarySize + k

          for l in range(1, boundarySize + 1):
            sumVar += data[toggle, baseI, baseJ, baseK - l] + \
                    data[toggle, baseI, baseJ, baseK + l] + \
                    data[toggle, baseI, baseJ - l, baseK] + \
                    data[toggle, baseI, baseJ + l, baseK] + \
                    data[toggle, baseI - l, baseJ, baseK] + \
                    data[toggle, baseI + l, baseJ, baseK]

          data[1 - toggle, baseI, baseJ, baseK] = sumVar / (6 * boundarySize)

  return data[1 - toggle, startDIdx:endDIdx, startHIdx:endHIdx, startWIdx:endWIdx]


def run(procHeight, procWidth, problemDepth, problemHeight, problemWidth,
        timeSteps, boundarySize):
  elfPaths = glob(f"{name}/bin/out_[0-9]*.elf")

  runner = CSELFRunner(elfPaths, cmaddr=args.cmaddr)

  # ISL map to indicate the PE that will produce the output wavelet, along with
  # the direction of the output wavelet
  output_port_map = f"{{out_tensor[idx=0:0] -> [PE[{procWidth-1},-1] -> index[idx]]}}"
  runner.add_output_tensor(8, output_port_map, np.uint32)

  # Simulate ELF file
  runner.connect_and_run()

  dataWidth = problemWidth // procWidth
  dataHeight = problemHeight // procHeight
  dataDepth = problemDepth

  bufferWidth = dataWidth + 2 * boundarySize
  bufferHeight = dataHeight + 2 * boundarySize
  bufferDepth = dataDepth + 2 * boundarySize

  shape = (2, bufferDepth, bufferHeight, bufferWidth)

  startWIdx = boundarySize
  endWIdx = startWIdx + dataWidth

  startHIdx = boundarySize
  endHIdx = startHIdx + dataHeight

  startDIdx = boundarySize
  endDIdx = startDIdx + dataDepth

  rect = ((1, 1), (procWidth, procHeight))
  results = runner.get_symbol_rect(rect, "data", np.float16)

  # The shape of results is (procWidth, procHeight, np.prod(shape)), but we need
  # this shape for comparing with the oracle. We swap the first two axes (width
  # for height), then reshape the contents to 'shape'.
  results = results.swapaxes(0, 1).reshape((procHeight, procWidth, *shape))

  resultIdx = timeSteps % 2
  subset = results[:, :, resultIdx, startDIdx:endDIdx, startHIdx:endHIdx, startWIdx:endWIdx]
  return subset.transpose(2, 0, 3, 1, 4).reshape(problemDepth, problemHeight, problemWidth)

# Parse the compile metadata
compile_data = None
with open(f"{name}/out.json", encoding="utf-8") as json_file:
  compile_data = json.load(json_file)
assert compile_data is not None
compile_params = compile_data["params"]
p_problemDepth = int(compile_params["problemDepth"])
p_problemHeight = int(compile_params["problemHeight"])
p_problemWidth = int(compile_params["problemWidth"])
p_computeHeight = int(compile_params["computeHeight"])
p_computeWidth = int(compile_params["computeWidth"])
p_timeSteps = int(compile_params["timeSteps"])
p_ghostCells = int(compile_params["ghostCells"])


computed = run(p_computeHeight, p_computeWidth, p_problemDepth, p_problemHeight,
               p_problemWidth, p_timeSteps, p_ghostCells)

oracle = computeOracle(p_problemDepth, p_problemHeight,
                       p_problemWidth, p_ghostCells, p_timeSteps)

np.testing.assert_allclose(oracle, computed, atol=0.001, rtol=0)
print("SUCCESS!")
