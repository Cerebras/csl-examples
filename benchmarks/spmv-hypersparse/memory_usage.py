# Copyright 2022 Cerebras Systems.
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


def memory_per_pe(max_local_nnz, max_local_nnz_cols, max_local_nnz_rows, local_in_vec_sz, local_out_vec_sz):
    '''
    // input matrix
    var mat_vals_buf = @zeros([max_local_nnz]f32);      // in matrix values (sparse): 4B
    var mat_rows_buf = @zeros([max_local_nnz]u16);      // in matrix relative row offsets: 2B
                                                    // need this in preprocessing: 2B
    var mat_col_idx_buf = @zeros([max_local_nnz_cols]u16);   // column idx of nnz cols (max possible size is nnz)
    var mat_col_loc_buf = @zeros([max_local_nnz_cols]u16);   // col location in mat_vals_buf and mat_rows_buf (max nnz)
    var mat_col_len_buf = @zeros([max_local_nnz_cols]u16);   // col length (nnz rows in a col)

    // input vector: for north-going and south-going trains
    // buffer storing data for tx
    var x_tx_buf = @zeros([local_vec_sz]f32);       // in vector values (dense): 4B
    // double buffers storing rx data
    var x_north_buf0 = @zeros([local_vec_sz]f32);   // in vector values (dense): 4B
    var x_south_buf0 = @zeros([local_vec_sz]f32);   // in vector values (dense): 4B
    var x_north_buf1 = @zeros([local_vec_sz]f32);   // in vector values (dense): 4B
    var x_south_buf1 = @zeros([local_vec_sz]f32);   // in vector values (dense): 4B

    // precomputed output vector (sparse format) local rows index information
    var y_rows_init_buf = @zeros([max_local_nnz_rows]u16);       // init -- this should not be modified

    // output vector (sparse): to store partial computed output vectors for north and south trains
    var y_vals_north_buf = @zeros([max_local_nnz_rows]f32);       // 4B
    var y_rows_north_buf = @zeros([max_local_nnz_rows]u16);       // 2B
    var y_vals_south_buf = @zeros([max_local_nnz_rows]f32);       // 4B
    var y_rows_south_buf = @zeros([max_local_nnz_rows]u16);       // 2B

    // buffers for east and west trains
    var y_vals_west_buf = @zeros([max_local_nnz_rows]f32);    // rx/tx vals on west-train during reduction (sparse): 4B
    var y_rows_west_buf = @zeros([max_local_nnz_rows]u16);    // rx/tx rows on west-train during reduction (sparse): 4B
    var y_vals_east_buf = @zeros([max_local_nnz_rows]f32);    // rx/tx vals on east-train during reduction (sparse): 4B
    var y_rows_east_buf = @zeros([max_local_nnz_rows]u16);    // rx/tx rows on east-train during reduction (sparse): 4B

    // final reduced local output vector (dense)
    var y_local_buf = @zeros([local_out_vec_sz]f32);    // 4B
    '''
    
    dtsz_u16 = np.uint16().itemsize     ## 2 bytes
    dtsz_f32 = np.float32().itemsize    ## 4 bytes
    
    ## input matrix in sparse format
    in_mat_mem = (dtsz_f32 + dtsz_u16) * max_local_nnz + 3 * dtsz_u16 * max_local_nnz_cols
    ## input vector in dense format
    in_vec_mem = 5 * dtsz_f32 * local_in_vec_sz                    ## 4 buffers + 1 tx
    ## partial output vector in sparse format
    sp_vec_init_mem = dtsz_u16 * max_local_nnz_rows ## init/precomputed rows data
    sp_vec_mem = 4 * ((dtsz_f32 + dtsz_u16) * max_local_nnz_rows)  ## 4 sets of buffers
    ## output vector in dense format
    out_vec_mem = dtsz_f32 * local_out_vec_sz
    
    return in_mat_mem + in_vec_mem + sp_vec_init_mem + sp_vec_mem + out_vec_mem
