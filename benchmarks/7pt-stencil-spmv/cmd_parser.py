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

"""command parser for bandwidthTest

   -m <int>     number of rows of the core rectangle
   -n <int>     number of columns of the core rectangle
   -k <int>     number of elements of local tensor
   --zDim <int>   number of elements to compute y=A*x
   --blockSize <int>  the size of temporary buffers for communication
   --latestlink   working directory
   --driver     path to CSL compiler
   --fabric-dims  fabric dimension of a WSE
   --cmaddr       IP address of a WSE
   --channels        number of I/O channels, 1 <= channels <= 16
   --width-west-buf  number of columns of the buffer in the west of the core rectangle
   --width-east-buf  number of columns of the buffer in the east of the core rectangle
   --compile-only    compile ELFs
   --run-only        run the test with precompiled binary
"""

import argparse
import os


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("-m", default=1, type=int, help="number of rows")
  parser.add_argument("-n", default=1, type=int, help="number of columns")
  parser.add_argument("--simulator", action="store_true", help="Runs on simulator")
  parser.add_argument("-k", default=1, type=int, help="size of local tensor, no less than 2")
  parser.add_argument(
      "--zDim",
      default=2,
      type=int,
      help="[0 zDim-1) is the domain of Laplacian",
  )
  parser.add_argument("--latestlink", help="folder to contain the log files (default: latest)")
  parser.add_argument("-d", "--driver", help="The path to the CSL compiler")
  parser.add_argument("--compile-only", help="Compile only", action="store_true")
  parser.add_argument("--fabric-dims", help="Fabric dimension, i.e. <W>,<H>")
  parser.add_argument("--cmaddr", help="CM address and port, i.e. <IP>:<port>")
  parser.add_argument("--run-only", help="Run only", action="store_true")
  parser.add_argument("--arch", help="wse2 or wse3. Default is wse2 when not supplied.")
  parser.add_argument("--width-west-buf", default=0, type=int, help="width of west buffer")
  parser.add_argument("--width-east-buf", default=0, type=int, help="width of east buffer")
  parser.add_argument(
      "--channels",
      default=1,
      type=int,
      help="number of I/O channels, between 1 and 16",
  )
  parser.add_argument(
      "--blockSize",
      default=2,
      type=int,
      help="the size of temporary buffers for communication",
  )

  args = parser.parse_args()

  logs_dir = "latest"
  if args.latestlink:
    logs_dir = args.latestlink

  dir_exist = os.path.isdir(logs_dir)
  if dir_exist:
    print(f"{logs_dir} already exists")
  else:
    print(f"create {logs_dir} to store log files")
    os.mkdir(logs_dir)

  if args.cmaddr is None:
    args.simulator = False

  return args, logs_dir
