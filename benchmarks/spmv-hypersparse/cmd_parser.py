# Copyright 2024 Cerebras Systems.
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
import tempfile
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--infile_mtx',
        help='the sparse matrix in MTX format',
        required=True
    )
    parser.add_argument("--simulator", action="store_true",
        help="Runs on simulator")
    parser.add_argument(
        '--num_pe_cols',
        type=int,
        help='width of the core rectangle',
        required=True
    )
    parser.add_argument(
        '--num_pe_rows',
        type=int,
        help='height of the core rectangle',
        required=True
    )
    parser.add_argument(
        "--fabric-dims",
        help="Fabric dimension, i.e. <W>,<H>"
    )
    parser.add_argument(
        "--compile-only",
        help="Compile only", action="store_true"
    )
    parser.add_argument(
        "--run-only",
        help="Run only", action="store_true"
    )
    parser.add_argument(
        "--width-west-buf",
        default=0,
        type=int,
        help="width of west buffer"
    )
    parser.add_argument(
        "--width-east-buf",
        default=0,
        type=int,
        help="width of east buffer"
    )
    parser.add_argument(
        "--channels",
        default=1,
        type=int,
        help="number of I/O channels, between 1 and 16"
    )
    parser.add_argument(
        "-d",
        "--driver",
        help="The path to the CSL compiler",
    )
    parser.add_argument(
        "--cmaddr",
        help="CM address and port, i.e. <IP>:<port>"
    )
    parser.add_argument(
        "--arch",
        help="wse2 or wse3. Default is wse2 when not supplied."
    )
    parser.add_argument(
        '--is_invec_one',
        help="input vector x is all one",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        '--is_weight_one',
        help="matrix A is from the given matrix or all one",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--latestlink",
        help="folder to contain the log files (default: latest)",
        default="latest"
    )

    args = parser.parse_args()

    if args.cmaddr is None:
        args.simulator = False

    return args
