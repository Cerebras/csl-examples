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


import os

import numpy as np

# name mapping between spmv kernel and this C code
#   C code           spmv kernel
# ----------------------------------
#  local_nzcols     local_nnzcols
#  local_nzrows     local_nnzrows
#  local_nnz        local_nnz
#  y_rows           y_rows_init_buf
#  A_colloc         mat_col_loc_buf
#  A_collen         mat_col_len_buf
#  A_colidx         mat_col_idx_buf
#  A_rows           mat_rows_buf
#  A_vals           mat_vals_buf
#
def preprocess(
    # A is nrows-by-ncols with nnz nonzeros
    nrows: int,
    ncols: int,
    nnz: int,
    # core rectangle of spmv is fabx-by-faby
    fabx: int,
    faby: int,
    # (csrRowPtr, csrColInd, csrVal) is the CSR representation
    csrRowPtr: np.ndarray,
    csrColInd: np.ndarray,
    csrVal: np.ndarray,
    # (cscColPtr, cscRowInd, cscVal) is the CSC representation
    cscColPtr: np.ndarray,
    cscRowInd: np.ndarray,
    cscVal: np.ndarray):
    """
    Given a spare matrix A of dimension nrows-by-ncols with nnz nonzeros
    and the dimension of core rectangle fabx-by-faby, parition the matrix
    A such that PE(px=j, py=i) contains the submatrix Aij with the 
    following quantities:
 
    local_nzrows: number of nonzero rows
    local_nzcols: number of nonzero columns
    local_nnz: number of nonzero elements
    y_rows[local_nzrows]: nonzero row index
    y_vals[local_nzrows]: not used
    A_colloc[local_nzcols]: prefix sum of A_collen, used to point to A_rows
    A_collen[local_nzcols]: A_collen[j] is number of nonzeros of j-th nonzero columns
    A_colidx[local_nzcols]: column index of nonzero columns
    A_rows[local_nnz]: position of row index of nonzeros in y_rows
    A_vals[local_nnz]: value of nonzeros

    """
    assert 0 == csrRowPtr[0], "CSR must be base-0"
    assert 0 == cscColPtr[0], "CSC must be base-0"
    assert nnz == csrRowPtr[nrows], "CSR has wrong nnz"
    assert nnz == cscColPtr[ncols], "CSC has wrong nnz"

    bx = int((ncols + fabx-1) / fabx) # number of columns of a block
    by = int((nrows + faby-1) / faby) # number of rows of a block
 
    local_nzrows = np.zeros((faby, fabx, 1), dtype = np.int32)
    local_nzcols = np.zeros((faby, fabx, 1), dtype = np.int32)
    local_nnz = np.zeros((faby, fabx, 1), dtype = np.int32)

    max_grid_dim = max(faby, fabx)
    counted = np.zeros(max_grid_dim, dtype = np.int32)

    # step 1: compute local_ncols and local_nnz
    counted[0:max_grid_dim] = -1 # invalid token 
    for col in range(ncols):
        check_token = col
        # col = col_b * bx + col_l
        # where col_b is the column block index
        #       col_l is local column index
        col_b = int(col / bx)
        col_l = col - col_b * bx
        start = cscColPtr[col]
        end = cscColPtr[col+1]
        for colidx in range(start, end):
            row = cscRowInd[colidx]
            # row = row_b * by + row_l
            # where row_b is the row block index
            #       row_l is local row index
            row_b = int(row / by)
            row_l = row - row_b * by
            local_nnz[(row_b, col_b)] += 1
            # Suppose Aij is block (row_b, col_b)
            # if |{Aij(i, col_l) != 0}| > 0, col_l is a nonzero column in Aij
            # we use counted[row_b] to count only once
            # if Aij(i1, col_l) and Aij(i2, col_l) are nonzero and i1 < i2,
            # only Aij(i1, col_l) adds local_nzcols[(row_b, col_b)]
            if (counted[row_b] != check_token):
                # Aij(row_l,col_l) is nonzero
                local_nzcols[(row_b, col_b)] += 1
                counted[row_b] = check_token

    # step 2: compute local_nrows
    counted[0:max_grid_dim] = -1 # invalid token
    for row in range(nrows):
        check_token = row
        # row = row_b * by + row_l
        row_b = int(row / by)
        row_l = row - row_b * by
        start = csrRowPtr[row]
        end = csrRowPtr[row+1]
        for colidx in range(start, end):
            col = csrColInd[colidx]
            # col = col_b * bx + col_l
            col_b = int(col / bx)
            col_l = col - col_b * bx
            # Suppose Aij is block (row_b, col_b)
            # if |{Aij(row_l, j) != 0}| > 0, row_l is a nonzero row in Aij
            # we use counted[col_b] to count only once
            # if Aij(row_l, j1) and Aij(row_l, j2) are nonzero and j1 < j2,
            # only Aij(row_l, j1) adds local_nzrows[(row_b, col_b)]
            if (counted[col_b] != check_token):
                # Aij(row_l,col_l) is nonzero
                local_nzrows[(row_b, col_b)] += 1
                counted[col_b] = check_token

    # step 3: compute maximum dimension of Aij
    max_local_nnz = max(local_nnz.ravel())
    max_local_nnz_cols = max(local_nzcols.ravel())
    max_local_nnz_rows = max(local_nzrows.ravel())

    assert max_local_nnz < np.iinfo(np.uint16).max,\
       "LOCAL NUMBER OF NONZEROS WILL OVERFLOW, TRY USING A LARGER FABRIC"
    assert max_local_nnz_cols < np.iinfo(np.uint16).max,\
       "LOCAL NUMBER OF NZCOLS WILL OVERFLOW, TRY USING A LARGER FABRIC"
    assert max_local_nnz_rows < np.iinfo(np.uint16).max,\
       "LOCAL NUMBER OF NZROWS WILL OVERFLOW, TRY USING A LARGER FABRIC"
    # no data overflows u16, we can convert the data to u16
    local_nnz = local_nnz.astype(np.uint16)
    local_nzrows = local_nzrows.astype(np.uint16)
    local_nzcols = local_nzcols.astype(np.uint16)

    #     spmv kernel                      actual storage in preprocess
    # ------------------------------------------------------------------
    # mat_vals_buf[max_local_nnz]           A_vals[local_nnz]
    # mat_rows_buf[max_local_nnz]           A_rows[local_nnz]
    # mat_col_loc_buf[max_local_nnz_cols]   A_colloc[local_nzcols]
    # mat_col_len_buf[max_local_nnz_cols]   A_collen[local_nzcols]
    # mat_col_idx_buf[max_local_nnz_cols]   A_colidx[local_nzcols]
    # y_rows_init_buf[max_local_nnz_rows]   y_rows[local_nzrows]
    #
    # To prepare the data for spmv, each PE allocates the maximum dimension
    # max_local_nnz, max_local_nnz_cols or max_local_nnz_rows
    A_vals = np.zeros((faby, fabx, max_local_nnz), dtype = np.float32)
    A_rows = np.zeros((faby, fabx, max_local_nnz), dtype = np.uint16)
    A_colloc = np.zeros((faby, fabx, max_local_nnz_cols), dtype = np.uint16)
    A_collen = np.zeros((faby, fabx, max_local_nnz_cols), dtype = np.uint16)
    A_colidx = np.zeros((faby, fabx, max_local_nnz_cols), dtype = np.uint16)
    y_rows = np.zeros((faby, fabx, max_local_nnz_rows), dtype = np.uint16)

    # step 4: compute y_rows
    local_pos = np.zeros((faby, fabx), dtype = np.int32)
    counted[0:max_grid_dim] = -1 # invalid token
    for row in range(nrows):
        check_token = row
        # row = row_b * by + row_l
        row_b = int(row / by)
        row_l = row - row_b * by
        start = csrRowPtr[row]
        end = csrRowPtr[row+1]
        for colidx in range(start, end):
            col = csrColInd[colidx]
            # col = col_b * bx + col_l
            col_b = int(col / bx)
            col_l = col - col_b * bx
            # Suppose Aij is block (row_b, col_b)
            # if |{Aij(row_l, j) != 0}| > 0, row_l is a nonzero row in Aij
            # we use counted[col_b] to count only once
            if (counted[col_b] != check_token):
                # Aij(row_l,col_l) is nonzero
                pos = local_pos[(row_b, col_b)]
                y_rows[(row_b, col_b, pos)] = row_l
                local_pos[(row_b, col_b)] = pos + 1 # advance to next nonzero row in Aij
                counted[col_b] = check_token

    # step 5: compute A_colloc, A_colidx, A_colen and A_rows
    #  y_rows is computed in step 4 because A_rows must be constructed by using y_rows

    # "local_pos" keeps track of the position of nonzero column in A_colidx
    local_pos = np.zeros((faby, fabx), dtype = np.int32)
    counted[0:max_grid_dim] = -1 # invalid token
    for col in range(ncols):
        check_token = col
        # col = col_b * bx + col_l
        # where col_b is the column block index
        #       col_l is local column index
        col_b = int(col / bx)
        col_l = col - col_b * bx
        start = cscColPtr[col]
        end = cscColPtr[col+1]
        for colidx in range(start, end):
            row = cscRowInd[colidx]
            val = cscVal[colidx]
            # row = row_b * by + row_l
            # where row_b is the row block index
            #       row_l is local row index
            row_b = int(row / by)
            row_l = row - row_b * by
            # Suppose Aij is block (row_b, col_b)
            # Aij(row_l,col_l) is nonzero
            if (counted[row_b] != check_token):
                # pos = position of nonzero column index in A_colidx and A_colen
                # A_collen[pos] is accumulated nonzero rows
                # A_colidx[pos] is the nonzero local column index
                pos = local_pos[(row_b, col_b)]
                # only record nonzero local column index once
                A_colidx[(row_b, col_b, pos)] = col_l
                # update A_colloc such that
                # A_colloc[0] = 0
                # A_colloc[j] = A_colloc[j-1] + A_colen[j-1]
                if (0 < pos):
                    A_colloc[(row_b, col_b, pos)] = A_colloc[(row_b, col_b, pos-1)] + A_collen[(row_b, col_b, pos-1)]
                local_pos[(row_b, col_b)] = pos + 1 # advance to next nonzero column in Aij
                counted[row_b] = check_token
            #else:
            #   "pos" is still current position of nonzero column index in A_colen

            # Remark: "pos" is well-defined because CSC is sorted in ascending order
            #   if col_l changes, then previous nonzero col_l is done
            #   When the loop enters 1st row_l of in Aij(:, col_l), it defines "pos"
            #   , the subsequent row_l in the same Aij(:, col_l) keeps the same "pos"
            #   When the loop exits Aij, A_collen and A_rows for Aij(:, col_l) is done
            #   When the loop enters Aij again, it re-starts the process for next nonzero
            #   col_l in Aij
            pos_start = A_colloc[(row_b, col_b, pos)] # position of 1st row index if Aij(:, col_l) in A_rows
            pos_rel_rowidx = A_collen[(row_b, col_b, pos)] # position of nonzero row index in A_rows
                                                           # corresponding to Aij(:, col_l)
            pos_rowidx = pos_rel_rowidx + pos_start
            # y_rows records distance(y_row.begin, find(y_rows.begin(), y_rows.end(), row_l))
            # spmv uses y_rows to store the result of outer-product of A*x
            y_rows_list = list(y_rows[(row_b, col_b)])
            A_rows[(row_b, col_b, pos_rowidx)] = y_rows_list.index(row_l)
            A_vals[(row_b, col_b, pos_rowidx)] = val
            A_collen[(row_b, col_b, pos)] = pos_rel_rowidx+1 # move to next nonzero Aij(row_l, col_l)


    matrix_info = {}
    matrix_info['nrows'] = nrows # number of rows of the matrix
    matrix_info['ncols'] = ncols # number of columns of the matrix
    matrix_info['nnz'] = nnz # number of nonzeros of the matrix
    matrix_info['max_local_nnz'] = max_local_nnz
    matrix_info['max_local_nnz_cols'] = max_local_nnz_cols
    matrix_info['max_local_nnz_rows'] = max_local_nnz_rows
    matrix_info['mat_vals_buf'] = A_vals
    matrix_info['mat_rows_buf'] = A_rows
    matrix_info['mat_col_loc_buf'] = A_colloc
    matrix_info['mat_col_len_buf'] = A_collen
    matrix_info['mat_col_idx_buf'] = A_colidx
    matrix_info['y_rows_init_buf'] = y_rows
    matrix_info['local_nnz'] = local_nnz
    matrix_info['local_nnz_cols'] = local_nzcols
    matrix_info['local_nnz_rows'] = local_nzrows

    return matrix_info
