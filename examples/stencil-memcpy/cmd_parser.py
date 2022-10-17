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

# This is not a real test, but a module that gets imported in other tests.

"""parse command line for sparse level routines

   -m <int>     number of rows of the matrix A
   -n <int>     number of columns of the matrix A
   --local_out_sz <int>  dimension of submatrix in tile approach,
                         or number of rows in non-tile approach
   --eps        tolerance
   --latestlink   working directory
   --debug      show A, x, and b
   --sdkgui     prepare data fro sdk gui, including source code
   --driver     path to CSL compiler
   --autocsl    use get_cslang_dir to find out the path of CSL

"""


import argparse


SIZE = 10
ZDIM = 10
ITERATIONS = 10
DX = 20


def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--name', help='the test name')
  parser.add_argument(
            '--zDim', type=int, help='size of the Z dimension', default=ZDIM
            )
  parser.add_argument(
            '--size', type=int, help='size of the domain in x and y dims', default=SIZE
            )

  parser.add_argument(
            '--skip-compile', action="store_true",
            help='Skip compilation of the code from python'
            )

  parser.add_argument(
            '--skip-run', action="store_true",
            help='Skip run of the code from python'
            )

  parser.add_argument(
            '--iterations',
            type=int,
            help='number of timesteps to simulate',
            default=ITERATIONS
            )

  parser.add_argument(
            '--dx',
            type=int,
            help='dx value (impacting the boundary)', default=DX
            )

  parser.add_argument(
            '--fabric_width',
            type=int,
            help='Width of the fabric we are compiling for',
            )

  parser.add_argument(
            '--fabric_height',
            type=int,
            help='Height of the fabric we are compiling for',
            )

  parser.add_argument(
            '-wout',
            '--wavefield_out',
            type=str,
            help='Read output wavefield and write npz file',
            default=None
            )
  parser.add_argument('--cmaddr', help='IP:port for CS system')

  parser.add_argument(
            "--debug",
            help="show A, x, and b", action="store_true"
            )

  parser.add_argument(
            "--enable-fifo",
            help="enable fifo in mux", action="store_true"
            )

  args = parser.parse_args()

  return args
