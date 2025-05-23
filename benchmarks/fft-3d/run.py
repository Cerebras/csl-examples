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
import time
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType  # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder  # pylint: disable=no-name-in-module
from cerebras.sdk import sdk_utils  # pylint: disable=no-name-in-module

def make_u48_array(b: np.ndarray):
  return b[..., 0] + (np.left_shift(b[..., 1], 16, dtype=int)) \
                   + (np.left_shift(b[..., 2], 32, dtype=int))

def arguments():
  parser = argparse.ArgumentParser(description="FFT parameters.")
  parser.add_argument(
    "-no_check",
    "--no-check-outcome",
    action="store_true",
    help="Do not validate outcome against numpy implementation",
  )
  parser.add_argument(
    "-n",
    "--name",
    required=True,
    default="out",
    help="Compile output directory.",
  )
  parser.add_argument(
    "-i",
    "--inverse",
    action="store_true",
    help="Compute the inverse FFT.",
    default=False,
  )
  parser.add_argument(
    "-r",
    "--real",
    action="store_true",
    help="Compute real FFT.",
    default=False,
  )
  parser.add_argument(
    "--norm",
    help="Normalization (0=backward, 1=ortho, 2=forward.",
    type=int,
    default=0,
  )
  parser.add_argument(
    "--save-output",
    action="store_true",
    help="Save result of FFT to .npy file.",
    default=False
  )
  parser.add_argument("--cmaddr", help="IP:port for CS system")
  return parser.parse_args()

args = arguments()

CHECK_RES = not args.no_check_outcome
REAL = args.real
NORM = np.int16(args.norm)
INVERSE = np.int16(args.inverse)
np_norm = "backward" if NORM == 0 else "forward" if NORM == 2 else "ortho"

# Parse the compile metadata
with open(f"{args.name}/out.json", encoding="utf-8") as json_file:
  compile_data = json.load(json_file)
compile_params = compile_data["params"]

N = int(compile_params["N"])
T = int(compile_params["NUM_PENCILS_PER_DIM"])
FP = int(compile_params["FP"])

# number of PEs per dimension is FFT size along dimension divided by
# number of FFT pencils per dimension
width = N // T

precision_type = np.dtype(np.float16 if FP == 0 else np.float32)
print(f"N: {N}, num_pencils_per_dim: {T}, kernel width: {width}, data type: {precision_type}")

# element has real and imaginary parts
ELEM_SIZE = 2

LOCAL_TENSOR_ELEM = T * T * N
LOCAL_TENSOR_LEN = LOCAL_TENSOR_ELEM * ELEM_SIZE

GLOBAL_TENSOR_ELEM = N * N * N
GLOBAL_TENSOR_LEN = GLOBAL_TENSOR_ELEM * ELEM_SIZE

# create twiddle factors
f_container = np.zeros(N, dtype=np.uint32)
f = sdk_utils.memcpy_view(f_container, precision_type)

Nh = N >> 1
exponent = np.pi * np.arange(Nh) / Nh
fI = np.sin(exponent).astype(precision_type)
fR = np.cos(exponent).astype(precision_type)
f[0::2] = fR
f[1::2] = fI

# Set the seed so that CI results are deterministic
np.random.seed(seed=7)

# Create random array if CHECK_RES set, or a fixed array if not
if CHECK_RES:
  random_array = np.random.random(GLOBAL_TENSOR_LEN).astype(precision_type)
else:
  random_array = np.arange(GLOBAL_TENSOR_LEN).astype(precision_type)

X_pre = random_array.reshape((N, N, ELEM_SIZE*N))

X_container = np.zeros((width, width, LOCAL_TENSOR_LEN), dtype=np.uint32)
X_res_container = np.zeros((width, width, LOCAL_TENSOR_LEN), dtype=np.uint32)

X = sdk_utils.memcpy_view(X_container, precision_type)
X_res = sdk_utils.memcpy_view(X_res_container, precision_type)

# Reshuffle input to expected order
# On each PE, pencils are interleaved
for x in range(N):
  for y in range(N):
    for z in range(N):
      offset = z * T * T + (y % T) * T + x % T
      X[y // T][x // T][offset*2] = X_pre[y][x][2*z]
      if REAL:
        X[y // T][x // T][offset*2+1] = 0
      else:
        X[y // T][x // T][offset*2+1] = X_pre[y][x][2*z+1]

#########################################
memcpy_dtype = MemcpyDataType.MEMCPY_32BIT if precision_type == np.float32 \
  else MemcpyDataType.MEMCPY_16BIT
memcpy_order = MemcpyOrder.ROW_MAJOR

# Create SdkRuntime and load and run program
runner = SdkRuntime(args.name, cmaddr=args.cmaddr, suppress_simfab_trace=True)
runner.load()
runner.run()

# Get device symbols for data array, twiddle factors, and timestamps
symbol_X = runner.get_id("X")
symbol_twiddle = runner.get_id("twiddle_array")
symbol_timestamps = runner.get_id("fft_time")

# Write twiddle factors 'f' to every PE
f_fill = np.full((width, width, N), f_container, dtype=np.uint32)
runner.memcpy_h2d(symbol_twiddle, f_fill.ravel(), 0, 0, width, width, N,
                  streaming=False, data_type=memcpy_dtype,
                  order=memcpy_order, nonblock=False)

tstart = time.time()
# Copy input data to device
runner.memcpy_h2d(symbol_X, X_container.ravel(), 0, 0, width, width, LOCAL_TENSOR_LEN,
                  streaming=False, data_type=memcpy_dtype,
                  order=memcpy_order, nonblock=False)
if REAL:
  if INVERSE:
    runner.launch("csfftExecC2R", NORM, nonblock=False)
  else:
    runner.launch("csfftExecR2C", NORM, nonblock=False)
else:
  runner.launch("csfftExecC2C", NORM, INVERSE, nonblock=False)

# Copy back result from device
runner.memcpy_d2h(X_res_container.ravel(), symbol_X, 0, 0, width, width, LOCAL_TENSOR_LEN,
                  streaming=False, data_type=memcpy_dtype,
                  order=memcpy_order, nonblock=False)
tstop = time.time()

print(f"Time to compute FFT and transfer result back: {tstop - tstart}s")

# Copy back timestamps from device
timestamps = np.zeros((width, width, 2), dtype=np.uint32)
runner.memcpy_d2h(timestamps.ravel(), symbol_timestamps, 0, 0, width, width, 2,
                  streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                  order=memcpy_order, nonblock=False)

# Compute worst PE time
timestamps = np.frombuffer(timestamps.tobytes(), dtype=np.uint16).reshape((width, width, 4))
u48cycles_array = make_u48_array(timestamps)
cycles = u48cycles_array[:, :].max()
cs2_freq = 850000000.0
compute_time = cycles / cs2_freq

print(f"Compute time on WSE: {compute_time}s, {cycles} cycles")

# Stop device program
runner.stop()

# Create result arrays to check result and write to file
result_array_pre = np.zeros((width, width, LOCAL_TENSOR_ELEM), dtype=complex)
result_array = np.zeros((N, N, N), dtype=complex)

# Create complex array out of real and imaginary parts
for row in range(width):
  for i in range(width):
    for j in range(LOCAL_TENSOR_ELEM):
      result_array_pre[row][i][j] = complex(
          X_res[row][i][j * 2], X_res[row][i][j * 2 + 1])

# Unshuffle result from N/T x N/T x N*T*T array to N x N x N array
for x in range(N):
  for y in range(N):
    for z in range(N):
      offset = z * T * T + (y % T) * T + x % T
      result_array[y][x][z] = result_array_pre[y // T][x // T][offset]

if args.save_output:
  np.save("result_array.npy", result_array)

# Reshape input array to match np.fft format
if REAL:
  random_array_sq = random_array.reshape((N, N, N, 2))[:, :, :, 0].reshape((N, N, N))
else:
  # For the float16 case, you must first cast array to float32. Otherwise
  # the view will interpret the four float16s as a single complex number.
  random_array_sq = random_array.astype(np.float32).reshape((N, N, N*2)).view(np.csingle)

# Compute numpy reference
if INVERSE:
  reference_array = np.fft.ifftn(random_array_sq, norm=np_norm)
else:
  reference_array = np.fft.fftn(random_array_sq, norm=np_norm)

# Check result against numpy reference
if CHECK_RES:
  # 16-bit calculation can have large relative errors for entries close to 0
  rtol = 0.5 if (precision_type == np.float16) else 0.01
  np.testing.assert_allclose(
    result_array, reference_array, rtol=rtol, atol=0)
  print("\nSUCCESS!")
