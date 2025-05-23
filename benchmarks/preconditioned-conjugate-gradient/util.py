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

import numpy as np
from scipy.sparse import coo_matrix


def COL_MAJOR(h, w, l, height, width, pe_length):
  assert 0 <= h < height
  assert 0 <= w < width
  assert 0 <= l < pe_length

  return h + w * height + l * height * width


def hwl_2_oned_colmajor(height: int, width: int, pe_length: int, A_hwl: np.ndarray, dtype):
  """
    Given a 3-D tensor A[height][width][pe_length], transform it to
    1D array by column-major
    """
  A_1d = np.zeros(height * width * pe_length, dtype)
  idx = 0
  for l in range(pe_length):
    for w in range(width):
      for h in range(height):
        A_1d[idx] = A_hwl[(h, w, l)]
        idx = idx + 1
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


#  y = Laplacian(x) for z=0,1,..,zDim-1
#
# The capacity of x and y can be bigger than zDim, but the physical domain is [0,zDim)
#
# The coordinates of physical domain are x,y,z.
# The physical layout of WSE is width, height.
# To avoid confusion, the kernel is written based on the layout of
# WSE, not physical domain of the application.
# For example, the user can match x-coordinate to x direction of
# WSE and y-coordinate to y-direction of WSE.
#              x-coord
#            +--------+
#    y-coord |        |
#            +--------+
#
# The stencil coefficients "stencil_coeff" can vary along x-y direction,
# but universal along z-direction. Each PE can have seven coefficents,
# west, east, south, north, bottom, top and center.
#
# Input:
#   stencil_coeff: size is (h,w,7)
#   x: size is (h,w,l)
# Output:
#   y: size is (h,w,l)
#
def laplacian(stencil_coeff, zDim, x, y):
  (height, width, pe_length) = x.shape
  assert zDim <= pe_length
  # y and x must have the same dimensions
  (m, n, k) = y.shape
  assert m == height
  assert n == width
  assert pe_length == k
  # stencil_coeff must be (h,w,7)
  (m, n, k) = stencil_coeff.shape
  assert m == height
  assert n == width
  assert k == 7

  #          North
  #           j
  #        +------+
  # West i |      | East
  #        +------+
  #          south
  for i in range(height):
    for j in range(width):
      for k in range(zDim):
        c_west = stencil_coeff[(i, j, 0)]
        c_east = stencil_coeff[(i, j, 1)]
        c_south = stencil_coeff[(i, j, 2)]
        c_north = stencil_coeff[(i, j, 3)]
        c_bottom = stencil_coeff[(i, j, 4)]
        c_top = stencil_coeff[(i, j, 5)]
        c_center = stencil_coeff[(i, j, 6)]

        west_buf = 0  # x[(i,-1,k)]
        if j > 0:
          west_buf = x[(i, j - 1, k)]
        east_buf = 0  # x[(i,w,k)]
        if j < width - 1:
          east_buf = x[(i, j + 1, k)]
        north_buf = 0
        # x[(-1,j,k)]
        if i > 0:
          north_buf = x[(i - 1, j, k)]
        south_buf = 0  # x[(h,j,k)]
        if i < height - 1:
          south_buf = x[(i + 1, j, k)]
        bottom_buf = 0  # x[(i,j,-1)]
        if k > 0:
          bottom_buf = x[(i, j, k - 1)]
        top_buf = 0  # x[(i,j,l)]
        if k < zDim - 1:
          top_buf = x[(i, j, k + 1)]
        center_buf = x[(i, j, k)]
        y[(i, j, k)] = (c_west * west_buf + c_east * east_buf + c_south * south_buf +
                        c_north * north_buf + c_bottom * bottom_buf + c_top * top_buf +
                        c_center * center_buf)


# Given a 7-point stencil, generate sparse matrix A.
# A is represented by CSR.
# The order of grids is column-major
def csr_7_pt_stencil(stencil_coeff, height, width, pe_length):
  # stencil_coeff must be (h,w,7)
  (m, n, k) = stencil_coeff.shape
  assert m == height
  assert n == width
  assert k == 7

  N = height * width * pe_length

  # each point has 7 coefficents at most
  cooRows = np.zeros(7 * N, np.int32)
  cooCols = np.zeros(7 * N, np.int32)
  cooVals = np.zeros(7 * N, np.float32)

  #          North
  #           j
  #        +------+
  # West i |      | East
  #        +------+
  #          south
  nnz = 0
  for i in range(height):
    for j in range(width):
      for k in range(pe_length):
        c_west = stencil_coeff[(i, j, 0)]
        c_east = stencil_coeff[(i, j, 1)]
        c_south = stencil_coeff[(i, j, 2)]
        c_north = stencil_coeff[(i, j, 3)]
        c_bottom = stencil_coeff[(i, j, 4)]
        c_top = stencil_coeff[(i, j, 5)]
        c_center = stencil_coeff[(i, j, 6)]

        center_idx = COL_MAJOR(i, j, k, height, width, pe_length)
        cooRows[nnz] = center_idx
        cooCols[nnz] = center_idx
        cooVals[nnz] = c_center
        nnz += 1
        # west_buf = 0 # x[(i,-1,k)]
        if j > 0:
          west_idx = COL_MAJOR(i, j - 1, k, height, width, pe_length)
          cooRows[nnz] = center_idx
          cooCols[nnz] = west_idx
          cooVals[nnz] = c_west
          nnz += 1
        # east_buf = 0  # x[(i,w,k)]
        if j < width - 1:
          east_idx = COL_MAJOR(i, j + 1, k, height, width, pe_length)
          cooRows[nnz] = center_idx
          cooCols[nnz] = east_idx
          cooVals[nnz] = c_east
          nnz += 1
        # north_buf = 0; # x[(-1,j,k)]
        if i > 0:
          north_idx = COL_MAJOR(i - 1, j, k, height, width, pe_length)
          cooRows[nnz] = center_idx
          cooCols[nnz] = north_idx
          cooVals[nnz] = c_north
          nnz += 1
        # south_buf = 0  # x[(h,j,k)]
        if i < height - 1:
          south_idx = COL_MAJOR(i + 1, j, k, height, width, pe_length)
          cooRows[nnz] = center_idx
          cooCols[nnz] = south_idx
          cooVals[nnz] = c_south
          nnz += 1
        # bottom_buf = 0 # x[(i,j,-1)]
        if k > 0:
          bottom_idx = COL_MAJOR(i, j, k - 1, height, width, pe_length)
          cooRows[nnz] = center_idx
          cooCols[nnz] = bottom_idx
          cooVals[nnz] = c_bottom
          nnz += 1
        # top_buf = 0    # x[(i,j,l)]
        if k < pe_length - 1:
          top_idx = COL_MAJOR(i, j, k + 1, height, width, pe_length)
          cooRows[nnz] = center_idx
          cooCols[nnz] = top_idx
          cooVals[nnz] = c_top
          nnz += 1

  A_coo = coo_matrix((cooVals, (cooRows, cooCols)), shape=(N, N))

  A_csr = A_coo.tocsr(copy=True)
  # sort column indices
  A_csr = A_csr.sorted_indices().astype(np.float32)
  assert A_csr.has_sorted_indices == 1, "Error: A is not sorted"

  return A_csr
