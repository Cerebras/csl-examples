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

""" test sparse matrix-vector multiplication

  This example aims at a hypersparse matrix with almost uniform distribution.
  The algorithm partitions the sparse matrix into 2D grids. The algorithm may
  fail if there exists one parition which has too many nonzeros to fit the
  memory capacity (48KB) of the PE.

  To obtain the best performance, the user may need to reorder the matrix such
  that the variation of the nonzeros of each parition is small.

  To run this example, the user has to provide a file of Matrix Market File
  format with 1-based index. For example, the user can reorder the matrix A by
  the permutation matrices P and Q, and writes P*A*Q^T to a file. One option is
  "util/analyze.cpp" which provides a load balancing algorithm.

  This example reads a MTX file, generates the vector x, partitions the matrix,
  and computes y = A*x.

  The framework is
  ---
       sync()  // synchronize all PEs to sample the reference clock
       tic()   // record start time
       spmv()  // compute y = A*x
       toc()   // record end time
  ---

  The tic() samples "time_start" and toc() samples "time_end". The sync() samples
  "time_ref" which is used to shift "time_start" and "time_end".
  The elapsed time is measured by
       cycles_send = max(time_end) - min(time_start)

  The overall runtime is computed via the following formula
       time_send = (cycles_send / 0.85) *1.e-3 us
  where a PE runs with clock speed 850MHz

  The spmv kernel performs y = A * x
  where A is m-by-n with nnz nonzeros

  The standard measurement counts the number of memory access of
       y[i] = sum{ Aij * xj : Aij is nonzero }
  - read Aij: nnz
  - read xj: nnz
  - write y[i]: m
  Total number of memory access: (2*nnz + m) f32

  Here is the list of parameters:
    --infile_mtx=<path to mtx file> contains the sparse matrix A
    --num_pe_rows=<int> specifies the height of the core rectangle
    --num_pe_cols=<int> specifies the width of the core rectangle
    --channels=<int> specifies the number of I/O channels, no bigger than 16

  How to compile and run
     To build a 5-by-4 core rectangle, we need to pass --num_pe_cols=5 --num_pe_rows=4
     Use the following command to compile
        python run.py --arch=wse2 --num_pe_cols=5 --num_pe_rows=4 --channels=1
           --driver=<path to cslc> --compile-only --infile_mtx=<path to mtx file>
     Use the following command to run
        python run.py --arch=wse2 --num_pe_cols=5 --num_pe_rows=4 --channels=1
           --is_weight_one --run-only --infile_mtx=<path to mtx file>
"""

import json
import math
import time
from typing import Optional

import numpy as np
import pandas as pd
from cmd_parser import parse_args
from memory_usage import memory_per_pe
from preprocess import preprocess
from scipy import sparse
from scipy.io import mmread

from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType, MemcpyOrder # pylint: disable=import-error,no-name-in-module
from cerebras.sdk.client import SdkCompiler, SdkRuntime # pylint: disable=import-error,no-name-in-module

hash_filename = "hash.json"


def make_u48(words):
  return words[0] + (words[1] << 16) + (words[2] << 32)


def hwl_to_oned_colmajor(height: int, width: int, pe_length: int, A_hwl: np.ndarray, dtype):
  """
    Given a 3-D tensor A[height][width][pe_length], transform it to
    1D array by column-major
    """
  if A_hwl.dtype == np.float32:
    A_1d = np.zeros(height * width * pe_length, dtype)
    idx = 0
    for l in range(pe_length):
      for w in range(width):
        for h in range(height):
          A_1d[idx] = A_hwl[(h, w, l)]
          idx = idx + 1
  elif A_hwl.dtype == np.uint16:
    assert dtype == np.uint32, "only support dtype = u32 if A is f16"
    A_1d = np.zeros(height * width * pe_length, dtype)
    idx = 0
    for l in range(pe_length):
      for w in range(width):
        for h in range(height):
          x = A_hwl[(h, w, l)]
          # x can be (np.float16, np.int16, np.uint16)
          # convert x to u16
          z = x.view(np.uint16)
          # zero extension of u16
          A_1d[idx] = np.uint32(z)
          idx = idx + 1
  else:
    raise RuntimeError(f"{type(A_hwl)} is not supported")

  return A_1d


def oned_to_hwl_colmajor(height: int, width: int, pe_length: int, A_1d: np.ndarray, dtype):
  """
    Given a 1-D tensor A_1d[height*width*pe_length], transform it to
    3-D tensor A[height][width][pe_length] by column-major
    """
  if dtype == np.float32:
    # only support f32 to f32
    assert A_1d.dtype == np.float32, "only support f32 to f32"
    A_hwl = np.reshape(A_1d, (height, width, pe_length), order="F")

  elif dtype == np.uint16:
    # only support u32 to u16 by dropping upper 16-bit
    assert A_1d.dtype == np.uint32, "only support u32 to u16"
    A_hwl = np.zeros((height, width, pe_length), dtype)
    idx = 0
    for l in range(pe_length):
      for w in range(width):
        for h in range(height):
          x = A_1d[idx]
          x = x & 0x0000FFFF  # drop upper 16-bit
          A_hwl[(h, w, l)] = np.uint16(x)
          idx = idx + 1
  else:
    raise RuntimeError(f"{dtype} is not supported")

  return A_hwl


def read_input_vector(IS_INVEC_1, vec_len):
  if IS_INVEC_1:
    return np.ones(vec_len).astype(np.float32)

  np.random.seed(0)
  return np.random.rand(vec_len).astype(np.float32)


# x is distributed into the core rectangle by the following steps
# step 1: distribute x into columns
#    vec_len_per_pe_col = ceil(vec_len / np_cols)
# step 2: distribute the column into PEs
#    vec_len_per_pe = ceil(vec_len_per_pe_col / np_rows)
#
# For example, if core rectangle is 2-by-2 and local_vec_sz is 13
#    Each column has vec_len_per_pe_col = ceil(13/2) = 7
#    The size of result is 7*2 = 14 which is bigger than local_vec_sz due to padding
#    Each PE has vec_len_per_pe = ceil(7/2) = 4
#
# If x is {1,2,3,4,5,6,7,8,9,10,11,12,13}, the core has
#          PE.x=0      PE.x=1
#    +-------------+-------------+
#    | {1,2,3,4}   | {8,9,10,11} | PE.y=0
#    +-------------+-------------+
#    | {5,6,7,x}   | {12,13,x,x} | PE.y=1
#    +-------------+-------------+
# column 0 has 7 elements, {1,2,3,4,5,6,7}
# column 1 has 6 elements, {8,9,10,11,12,13}
#
# The symbol x is DON'T CARE
#
def dist_x_to_hwl(ncols, x, local_vec_sz, np_cols, np_rows):
  # core rectangle is np_cols-by-np_rows
  #            np_cols
  #         +----------+
  # np_rows |  core    |
  #         +----------+
  # input vector is distributed into columns, then distributed into rows

  vec_len = ncols
  vec_len_per_pe_col = math.ceil(vec_len / np_cols)
  vec_len_per_pe = math.ceil(vec_len_per_pe_col / np_rows)
  assert vec_len_per_pe == local_vec_sz

  pad_len_per_pe_col = (vec_len_per_pe * np_rows) - vec_len_per_pe_col

  pad_len = (vec_len_per_pe_col * np_cols) - vec_len
  # invec = [x, ones(pad_len)]
  invec = np.copy(x)
  ## BIG NOTE: Since this is input vector, padding needs to be 1s
  if pad_len > 0:
    invec = np.append(invec, np.ones(pad_len))

  x_hwl = np.zeros((np_rows, np_cols, vec_len_per_pe), x.dtype)
  ## now this is equally divided into np_cols
  for col in range(np_cols):
    ## get the slice for this col and append padding
    invec_col = invec[col * vec_len_per_pe_col:(col + 1) * vec_len_per_pe_col]
    if pad_len_per_pe_col > 0:
      invec_col = np.append(invec_col, np.ones(pad_len_per_pe_col)).astype(x.dtype)
    ## now this is equally divided into np_rows
    for row in range(np_rows):
      ## get the slice for this row
      data = invec_col[row * vec_len_per_pe:(row + 1) * vec_len_per_pe]
      x_hwl[(row, col)] = data

  return x_hwl


# The dimension of out_vec is h-by-w-by-l
# h = np_rows is the height of the core
# w = np_cols is the width of the core
# l = local_out_vec_sz is the size of local vector
#
# The out_vec_sz is the length of y = A*x
#
# y is distributed into the core rectangle by the following steps
# step 1: distribute y into rows
#    vec_len_per_pe_row = math.ceil(out_vec_sz / np_rows)
# step 2: distribute the row into PEs
#    vec_len_per_pe = math.ceil(vec_len_per_pe_row / np_cols)
#
# If out_vec_sz is smaller than (vec_len_per_pe_row*np_rows), padding is added
#
# The function unpad_3d_to_1d returns a result of size (vec_len_per_pe_row*np_rows)
#
# For example, if core rectangle is 2-by-2 and out_vec_sz is 13
#    Each row has vec_len_per_pe_row = ceil(13/2) = 7
#    The size of result is 7*2 = 14 which is bigger than out_vec_sz due to padding
#    Each PE has vec_len_per_pe = ceil(7/2) = 4
#
# If y is {1,2,3,4,5,6,7,8,9,10,11,12,13}, the core has
#          PE.x=0      PE.x=1
#    +-------------+-------------+
#    | {1,2,3,4}   | {5,6,7,x}   | PE.y=0
#    +-------------+-------------+
#    | {8,9,10,11} | {12,13,x,x} | PE.y=1
#    +-------------+-------------+
# row 0 has 7 elements, {1,2,3,4,5,6,7
# row 1 has 6 elements, {8,9,10,11,12,13}
#
# The symbol x is DON'T CARE
#
def unpad_3d_to_1d(out_vec_sz, out_vec):
  assert out_vec.ndim == 3, "y must be a 3-d tensor of the form h-by-w-by-l"
  (height, width, local_out_vec_sz) = out_vec.shape
  # core rectangle is np_cols-by-np_rows
  #            np_cols
  #         +----------+
  # np_rows |  core    |
  #         +----------+
  np_rows = height
  np_cols = width

  vec_len_per_pe_row = math.ceil(out_vec_sz / np_rows)
  vec_len_per_pe = math.ceil(vec_len_per_pe_row / np_cols)
  # check if local_out_vec_sz = math.ceil(math.ceil(out_vec_sz / np_rows) / np_cols)
  assert vec_len_per_pe == local_out_vec_sz

  # result includes the padding
  #    y = result[0:out_vec_sz]
  # clear result to avoid bogus value outside the range [0, out_vec_sz)
  result = np.zeros(vec_len_per_pe_row * np_rows, dtype=np.float32)
  # tmp_buf contains the padding one row PEs
  # tmp_buf gathers data of a whole row PE
  tmp_buf = np.empty(vec_len_per_pe * np_cols, dtype=np.float32)
  for row in range(np_rows):
    low_idx = row * vec_len_per_pe_row
    high_idx = low_idx + vec_len_per_pe_row
    # gather data into tmp_buf
    for col in range(np_cols):
      start = col * vec_len_per_pe
      end = start + vec_len_per_pe
      tmp_buf[start:end] = out_vec[(row, col)]
    result[low_idx:high_idx] = tmp_buf[0:vec_len_per_pe_row]
  return result


def verify_result(ref, res):
  print("Comparing result with reference...")
  abs_diff = np.sum(abs(ref - res))
  abs_rel = abs_diff / len(ref)
  print(f"reference[{len(ref)}]: \n{ref}")
  print(f"result   [{len(res)}]: \n{res}")
  print(f"[[ Absolute diff: {abs_diff} ]]")
  print(f"[[ Average diff : {abs_rel} ]]")
  atol = 1e-8
  rtol = 1e-5
  is_correct = np.allclose(ref, res, rtol, atol)
  result = "PASS" if is_correct else "FAIL"
  print(f"[[ Result within tolerance {atol}: {result} ]]")
  print(f"[[ Result within tolerance {atol}: {result} ]]")
  if not is_correct:
    unequal = ~np.isclose(ref, res)
    unequal_idx = list(np.where(unequal))
    mismatches = list(zip(ref[tuple(unequal_idx)], res[tuple(unequal_idx)]))
    df = pd.DataFrame(mismatches, columns=["reference", "result"], index=unequal_idx)
    print(f"{df}")


# y = A*x
# where A is nrows-by-ncols, represented by a CSR triplet
def generate_reference(nrows, ncols, csrRowPtr, csrColInd, csrVal, x):
  assert ncols == len(x), "the dimension of x does not match the dimension of A"
  mat = sparse.csr_matrix((csrVal, csrColInd, csrRowPtr), shape=(nrows, ncols))
  y = mat.dot(np.array(x).transpose())
  return y


def timing_analysis(height, width, nnz, time_memcpy_hwl, time_ref_hwl):
  time_start = np.zeros((height, width)).astype(int)
  time_end = np.zeros((height, width)).astype(int)
  word = np.zeros(3).astype(np.uint16)
  for w in range(width):
    for h in range(height):
      word[0] = time_memcpy_hwl[(h, w, 0)]
      word[1] = time_memcpy_hwl[(h, w, 1)]
      word[2] = time_memcpy_hwl[(h, w, 2)]
      time_start[(h, w)] = make_u48(word)
      word[0] = time_memcpy_hwl[(h, w, 3)]
      word[1] = time_memcpy_hwl[(h, w, 4)]
      word[2] = time_memcpy_hwl[(h, w, 5)]
      time_end[(h, w)] = make_u48(word)

  # time_ref = reference clock
  time_ref = np.zeros((height, width)).astype(int)
  word = np.zeros(3).astype(np.uint16)
  for w in range(width):
    for h in range(height):
      word[0] = time_ref_hwl[(h, w, 0)]
      word[1] = time_ref_hwl[(h, w, 1)]
      word[2] = time_ref_hwl[(h, w, 2)]
      time_ref[(h, w)] = make_u48(word)

  # adjust the reference clock by the propagation delay
  # the right-bottom PE signals other PEs, the propagation delay is
  #     (h-1) - py + (w-1) - px
  for py in range(height):
    for px in range(width):
      time_ref[(py, px)] = time_ref[(py, px)] - ((width + height - 2) - (px + py))

  # shift time_start and time_end by time_ref
  time_start = time_start - time_ref
  time_end = time_end - time_ref

  # cycles_send = time_end[(h,w)] - time_start[(h,w)]
  # 850MHz --> 1 cycle = (1/0.85) ns = (1/0.85)*1.e-3 us
  # time_send = (cycles_send / 0.85) *1.e-3 us
  #
  # The spmv kernel performs y = A * x
  #   y[i] = sum{ Aij * xj : Aij is nonzero }
  # where A is m-by-n with nnz nonzeros
  #
  # We use the following standard measurement
  # - read Aij: nnz
  # - read xj: nnz
  # - write y[i]: m
  # Total number of wavelets: (2*nnz + m)
  #
  wvlts = 2 * nnz + height
  min_time_start = time_start.min()
  max_time_end = time_end.max()
  cycles_send = max_time_end - min_time_start
  time_send = (cycles_send / 0.85) * 1.0e-3
  bandwidth = (wvlts * 4) / time_send
  print(f"cycles_send = {cycles_send} cycles")
  print(f"time_send = {time_send} us")
  print(f"bandwidth = {bandwidth} MB/S ")


def csl_compile_core(
    csl_path: str,  # path to CSL files
    file_config: str,
    elf_dir: str,
    fabric_width: int,
    fabric_height: int,
    core_fabric_offset_x: int,  # fabric-offsets of the core
    core_fabric_offset_y: int,
    arch: Optional[str],
    ncols: int,
    nrows: int,
    np_cols: int,
    np_rows: int,
    max_local_nnz: int,
    max_local_nnz_cols: int,
    max_local_nnz_rows: int,
    local_vec_sz: int,
    local_out_vec_sz: int,
    out_pad_start_idx: int,
    channels: int,
    width_west_buf: int,
    width_east_buf: int,
):
  with SdkCompiler() as compiler:
    args = []
    args.append(f"--fabric-dims={fabric_width},{fabric_height}")  # options
    args.append(f"--fabric-offsets={core_fabric_offset_x},{core_fabric_offset_y}")  # options
    args.append(f"--params=ncols:{ncols}")  # options
    args.append(f"--params=nrows:{nrows}")  # options
    args.append(f"--params=pcols:{np_cols}")  # options
    args.append(f"--params=prows:{np_rows}")  # options
    args.append(f"--params=max_local_nnz:{max_local_nnz}")  # options
    args.append(f"--params=max_local_nnz_cols:{max_local_nnz_cols}")  # options
    args.append(f"--params=max_local_nnz_rows:{max_local_nnz_rows}")  # options
    args.append(f"--params=local_vec_sz:{local_vec_sz}")  # options
    args.append(f"--params=local_out_vec_sz:{local_out_vec_sz}")  # options
    args.append(f"--params=y_pad_start_row_idx:{out_pad_start_idx}")  # options

    args.append(f"-o={elf_dir}")
    if arch is not None:
      args.append(f"--arch={arch}")
    args.append("--memcpy")
    args.append(f"--channels={channels}")
    args.append(f"--width-west-buf={width_west_buf}")
    args.append(f"--width-east-buf={width_east_buf}")

    args_str = " ".join(args)
    hashstr = compiler.compile(csl_path, file_config, args_str)
    print("compile artifact (csl_hash/oname):", hashstr)
    return hashstr


# How to compile:
#  python run.py --arch=wse2 --num_pe_cols=4 --num_pe_rows=4 --channels=1 \
#    --width-west-buf=0 --width-east-buf=0 --is_weight_one --compile-only \
#    --infile_mtx=data/rmat4.4x4.lb.mtx
#
# How to run:
#  python run.py --arch=wse2 --num_pe_cols=4 --num_pe_rows=4 --channels=1 \
#    --width-west-buf=0 --width-east-buf=0 --is_weight_one --run-only \
#    --infile_mtx=data/rmat4.4x4.lb.mtx
#
def main():
  """Main method to run the example code."""

  args = parse_args()

  width_west_buf = args.width_west_buf
  width_east_buf = args.width_east_buf
  channels = args.channels
  assert channels <= 16, "only support up to 16 I/O channels"
  assert channels >= 1, "number of I/O channels must be at least 1"

  print(f"width_west_buf = {width_west_buf}")
  print(f"width_east_buf = {width_east_buf}")
  print(f"channels = {channels}")

  dirname = args.latestlink

  # core rectangle is np_cols-by-np_rows
  np_cols = args.num_pe_cols
  np_rows = args.num_pe_rows
  IS_INVEC_1 = args.is_invec_one

  width = np_cols
  height = np_rows
  print(f"width = {width}, height = {height}")

  start = time.time()
  infile_mtx = args.infile_mtx
  print(f"infile_mtx = {infile_mtx}")

  A_coo = mmread(infile_mtx)
  # the CSR format is 0-based
  A_csr = A_coo.tocsr(copy=True)
  # sort column indices
  A_csr = A_csr.sorted_indices().astype(np.float32)
  assert A_csr.has_sorted_indices == 1, "Error: A is not sorted"

  [nrows, ncols] = A_csr.shape
  nnz = A_csr.nnz

  print(f"Load matrix A, {nrows}-by-{ncols} with {nnz} nonzeros")

  if not args.is_weight_one:
    print("WARNING: reset the matrix with random values")
    np.random.seed(123)
    (A_csr.data)[0:nnz] = np.random.rand(nnz).astype(np.float32)

  csrRowPtr = A_csr.indptr
  csrColInd = A_csr.indices
  csrVal = A_csr.data

  A_csc = A_csr.tocsc(copy=True)
  # sort row indices
  A_csc = A_csc.sorted_indices().astype(np.float32)
  assert A_csc.has_sorted_indices == 1, "Error: A is not sorted"

  cscColPtr = A_csc.indptr
  cscRowInd = A_csc.indices
  cscVal = A_csc.data

  matrix_info = preprocess(
      # A is nrows-by-ncols with nnz nonzeros
      nrows,
      ncols,
      nnz,
      # core rectangle is fabx-by-faby
      np_cols,
      np_rows,
      # (csrRowPtr, csrColInd, csrVal) is the CSR representation
      csrRowPtr,
      csrColInd,
      # (cscColPtr, cscRowInd, cscVal) is the CSC representation
      cscColPtr,
      cscRowInd,
      cscVal,
  )

  end = time.time()
  print(f"prepare the structure for spmv kernel: {end-start}s", flush=True)

  max_local_nnz = matrix_info["max_local_nnz"]
  max_local_nnz_cols = matrix_info["max_local_nnz_cols"]
  max_local_nnz_rows = matrix_info["max_local_nnz_rows"]
  mat_vals_buf = matrix_info["mat_vals_buf"]
  mat_rows_buf = matrix_info["mat_rows_buf"]
  mat_col_idx_buf = matrix_info["mat_col_idx_buf"]
  mat_col_loc_buf = matrix_info["mat_col_loc_buf"]
  mat_col_len_buf = matrix_info["mat_col_len_buf"]
  y_rows_init_buf = matrix_info["y_rows_init_buf"]
  local_nnz = matrix_info["local_nnz"]
  local_nnz_cols = matrix_info["local_nnz_cols"]
  local_nnz_rows = matrix_info["local_nnz_rows"]

  x_ref = read_input_vector(IS_INVEC_1, ncols)

  # core rectangle is np_cols-by-np_rows
  #            np_cols
  #         +----------+
  # np_rows |  core    |
  #         +----------+
  # input vector is distributed into columns, then distributed into rows
  # output vector is distributed into rows, then distributed into columns
  local_vec_sz = math.ceil(math.ceil(ncols / np_cols) / np_rows)
  local_out_vec_sz = math.ceil(math.ceil(nrows / np_rows) / np_cols)

  x_tx_buf = dist_x_to_hwl(ncols, x_ref, local_vec_sz, np_cols, np_rows)

  print("Generating reference y = A*x ...")
  y_ref = generate_reference(nrows, ncols, csrRowPtr, csrColInd, csrVal, x_ref)

  mem_use_per_pe = memory_per_pe(
      max_local_nnz,
      max_local_nnz_cols,
      max_local_nnz_rows,
      local_vec_sz,
      local_out_vec_sz,
  )
  print(
      f"Total memory use per PE = {mem_use_per_pe} bytes = {mem_use_per_pe / 1024} KB",
      flush=True,
  )
  assert (mem_use_per_pe < 46 * 1024), "exceed maximum memory capacity, increase the core rectangle"

  # fabric-offsets = 1,1
  fabric_offset_x = 1
  fabric_offset_y = 1
  # starting point of the core rectangle = (core_fabric_offset_x, core_fabric_offset_y)
  # memcpy framework requires 3 columns at the west of the core rectangle
  # memcpy framework requires 2 columns at the east of the core rectangle
  core_fabric_offset_x = fabric_offset_x + 3 + width_west_buf
  core_fabric_offset_y = fabric_offset_y
  # (min_fabric_width, min_fabric_height) is the minimal dimension to run the app
  min_fabric_width = core_fabric_offset_x + width + 2 + 1 + width_east_buf
  min_fabric_height = core_fabric_offset_y + height + 1

  fabric_width = 0
  fabric_height = 0
  if args.fabric_dims:
    w_str, h_str = args.fabric_dims.split(",")
    fabric_width = int(w_str)
    fabric_height = int(h_str)

  if fabric_width == 0 or fabric_height == 0:
    fabric_width = min_fabric_width
    fabric_height = min_fabric_height

  assert fabric_width >= min_fabric_width
  assert fabric_height >= min_fabric_height

  print(f"fabric_width = {fabric_width}, fabric_height = {fabric_height}")
  print(
      f"core_fabric_offset_x = {core_fabric_offset_x}, "
      f"core_fabric_offset_y = {core_fabric_offset_y}"
  )

  # prepare the simulation
  print("store ELFs and log files in the folder ", dirname)

  # layout of a rectangle
  code_csl = "layout.csl"

  ## calculate the output vector padding info
  out_vec_len_per_pe_row = math.ceil(nrows / np_rows)
  out_pad_start_idx = out_vec_len_per_pe_row

  csl_path = "./src"

  if args.compile_only:
    print(
        "WARNING: compile the code, don't run SdkRuntime because "
        "the server is down after the compilation"
    )
    start = time.time()
    hashstr = csl_compile_core(
        csl_path,
        code_csl,
        dirname,
        fabric_width,
        fabric_height,
        core_fabric_offset_x,  # fabric-offsets of the core
        core_fabric_offset_y,
        args.arch,
        ncols,  # m, number of rows of the matrix
        nrows,  # n, number of columns of the matrix
        np_cols,  # width
        np_rows,  # height
        max_local_nnz,
        max_local_nnz_cols,
        max_local_nnz_rows,
        local_vec_sz,
        local_out_vec_sz,
        out_pad_start_idx,
        channels,
        width_west_buf,
        width_east_buf,
    )
    end = time.time()
    print(f"Compilation done in {end-start}s", flush=True)
    print(f"dump artifact name to file {hash_filename}")
    with open(hash_filename, "w", encoding="utf-8") as write_file:
      json.dump(hashstr, write_file)
    print("COMPILE ONLY: EXIT")
    return

  print(f"load artifact name from file {hash_filename}")
  with open(hash_filename, "r", encoding="utf-8") as f:
    hashstr = json.load(f)

  start = time.time()
  with SdkRuntime(hashstr, simulator=args.simulator) as runner:

    sym_mat_vals_buf = runner.get_id("mat_vals_buf")
    sym_x_tx_buf = runner.get_id("x_tx_buf")
    sym_y_local_buf = runner.get_id("y_local_buf")

    sym_mat_rows_buf = runner.get_id("mat_rows_buf")
    sym_mat_col_idx_buf = runner.get_id("mat_col_idx_buf")
    sym_mat_col_loc_buf = runner.get_id("mat_col_loc_buf")
    sym_mat_col_len_buf = runner.get_id("mat_col_len_buf")
    sym_y_rows_init_buf = runner.get_id("y_rows_init_buf")
    sym_local_nnz = runner.get_id("local_nnz")
    sym_local_nnz_cols = runner.get_id("local_nnz_cols")
    sym_local_nnz_rows = runner.get_id("local_nnz_rows")
    sym_time_buf_u16 = runner.get_id("time_buf_u16")
    sym_time_ref_u16 = runner.get_id("time_ref_u16")

    # load() and run() are called by client.Sdkruntime.__enter__
    # runner.load()
    # runner.run()

    print("step 1: enable tsc counter to sample the clock")
    runner.launch("f_enable_tsc", nonblock=True)

    print("step 2: copy the structure of A and vector x to the device")
    # 1. mat_vals_buf[max_local_nnz], type = f32
    mat_vals_buf_1d = hwl_to_oned_colmajor(height, width, max_local_nnz, mat_vals_buf, np.float32)
    runner.memcpy_h2d(
        sym_mat_vals_buf,
        mat_vals_buf_1d,
        0,
        0,
        width,
        height,
        max_local_nnz,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        order=MemcpyOrder.COL_MAJOR,
        nonblock=True,
    )

    # 2: x_tx_buf[local_vec_sz], type = f32
    x_tx_buf_1d = hwl_to_oned_colmajor(height, width, local_vec_sz, x_tx_buf, np.float32)
    runner.memcpy_h2d(
        sym_x_tx_buf,
        x_tx_buf_1d,
        0,
        0,
        width,
        height,
        local_vec_sz,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        order=MemcpyOrder.COL_MAJOR,
        nonblock=True,
    )

    # 3: mat_rows_buf[max_local_nnz], type = u16
    mat_rows_buf_1d = hwl_to_oned_colmajor(height, width, max_local_nnz, mat_rows_buf, np.uint32)
    runner.memcpy_h2d(
        sym_mat_rows_buf,
        mat_rows_buf_1d,
        0,
        0,
        width,
        height,
        max_local_nnz,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_16BIT,
        order=MemcpyOrder.COL_MAJOR,
        nonblock=True,
    )

    # 4: mat_col_idx_buf[max_local_nnz_cols], type = u16
    mat_col_idx_buf_1d = hwl_to_oned_colmajor(height, width, max_local_nnz_cols, mat_col_idx_buf,
                                              np.uint32)
    runner.memcpy_h2d(
        sym_mat_col_idx_buf,
        mat_col_idx_buf_1d,
        0,
        0,
        width,
        height,
        max_local_nnz_cols,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_16BIT,
        order=MemcpyOrder.COL_MAJOR,
        nonblock=True,
    )

    # 5: mat_col_loc_buf[max_local_nnz_cols], type = u16
    mat_col_loc_buf_1d = hwl_to_oned_colmajor(height, width, max_local_nnz_cols, mat_col_loc_buf,
                                              np.uint32)
    runner.memcpy_h2d(
        sym_mat_col_loc_buf,
        mat_col_loc_buf_1d,
        0,
        0,
        width,
        height,
        max_local_nnz_cols,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_16BIT,
        order=MemcpyOrder.COL_MAJOR,
        nonblock=True,
    )

    # 6: mat_col_len_buf[max_local_nnz_cols], type = u16
    mat_col_len_buf_1d = hwl_to_oned_colmajor(height, width, max_local_nnz_cols, mat_col_len_buf,
                                              np.uint32)
    runner.memcpy_h2d(
        sym_mat_col_len_buf,
        mat_col_len_buf_1d,
        0,
        0,
        width,
        height,
        max_local_nnz_cols,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_16BIT,
        order=MemcpyOrder.COL_MAJOR,
        nonblock=True,
    )

    # 7: y_rows_init_buf[max_local_nnz_rows], type = u16
    y_rows_init_buf_1d = hwl_to_oned_colmajor(height, width, max_local_nnz_rows, y_rows_init_buf,
                                              np.uint32)
    runner.memcpy_h2d(
        sym_y_rows_init_buf,
        y_rows_init_buf_1d,
        0,
        0,
        width,
        height,
        max_local_nnz_rows,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_16BIT,
        order=MemcpyOrder.COL_MAJOR,
        nonblock=True,
    )

    # 8: local_nnz, type = u16
    local_nnz_1d = hwl_to_oned_colmajor(height, width, 1, local_nnz, np.uint32)
    runner.memcpy_h2d(
        sym_local_nnz,
        local_nnz_1d,
        0,
        0,
        width,
        height,
        1,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_16BIT,
        order=MemcpyOrder.COL_MAJOR,
        nonblock=True,
    )

    # 9: local_nnz_cols, type = u16
    local_nnz_cols_1d = hwl_to_oned_colmajor(height, width, 1, local_nnz_cols, np.uint32)
    runner.memcpy_h2d(
        sym_local_nnz_cols,
        local_nnz_cols_1d,
        0,
        0,
        width,
        height,
        1,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_16BIT,
        order=MemcpyOrder.COL_MAJOR,
        nonblock=True,
    )

    # 10: local_nnz_rows, type = u16
    local_nnz_rows_1d = hwl_to_oned_colmajor(height, width, 1, local_nnz_rows, np.uint32)
    runner.memcpy_h2d(
        sym_local_nnz_rows,
        local_nnz_rows_1d,
        0,
        0,
        width,
        height,
        1,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_16BIT,
        order=MemcpyOrder.COL_MAJOR,
        nonblock=True,
    )

    print("step 3: sync all PEs to sample the reference clock")
    runner.launch("f_sync", np.int16(1), nonblock=False)

    print("step 4: tic() records time_start")
    runner.launch("f_tic", nonblock=True)

    print("step 5: spmv")
    runner.launch("f_spmv", nonblock=False)

    print("step 5: toc() records time_end")
    runner.launch("f_toc", nonblock=False)

    print("step 6: prepare (time_start, time_end)")
    runner.launch("f_memcpy_timestamps", nonblock=False)

    print("step 7: fetch the timing time_buf_u16[6] = (time_start, time_end), type = u16")
    time_memcpy_hwl_1d = np.zeros(height * width * 6, np.uint32)
    runner.memcpy_d2h(
        time_memcpy_hwl_1d,
        sym_time_buf_u16,
        0,
        0,
        width,
        height,
        6,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_16BIT,
        order=MemcpyOrder.COL_MAJOR,
        nonblock=False,
    )
    time_memcpy_hwl = oned_to_hwl_colmajor(height, width, 6, time_memcpy_hwl_1d, np.uint16)

    print("step 8: fetch the output vector y of type f32")
    y_1d = np.zeros(height * width * local_out_vec_sz, np.float32)
    runner.memcpy_d2h(
        y_1d,
        sym_y_local_buf,
        0,
        0,
        width,
        height,
        local_out_vec_sz,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        order=MemcpyOrder.COL_MAJOR,
        nonblock=False,
    )

    print("step 9: prepare reference clock")
    runner.launch("f_reference_timestamps", nonblock=False)

    print("step 10: D2H reference clock")
    time_ref_1d = np.zeros(height * width * 3, np.uint32)
    runner.memcpy_d2h(
        time_ref_1d,
        sym_time_ref_u16,
        0,
        0,
        width,
        height,
        3,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_16BIT,
        order=MemcpyOrder.COL_MAJOR,
        nonblock=False,
    )
    time_ref_hwl = oned_to_hwl_colmajor(height, width, 3, time_ref_1d, np.uint16)

    # stop() is called by client.Sdkruntime.__exit__
    # runner.stop()

  end = time.time()
  print(f"*** Run done in {end-start}s")

  timing_analysis(height, width, nnz, time_memcpy_hwl, time_ref_hwl)

  # The output y_wse distributed into nrows-by-ncols PEs
  y_wse = np.reshape(y_1d, (height, width, local_out_vec_sz), order="F")
  # y_wse is packed into 1d vector with zero padding
  y_wse = unpad_3d_to_1d(nrows, y_wse)
  # remove padding of y_wse because y_ref has no padding
  verify_result(y_ref, y_wse[0:nrows])

  # dump the device memory via debug tool
  if args.simulator:
    print(f"time_ref_hwl = \n{time_ref_hwl}")

if __name__ == "__main__":
  main()
