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

# This is not a real test, but a module that gets imported in other tests.

"""command parser for broadcast

   -m <int>      number of rows of the core rectangle
   -n <int>      number of columns of the core rectangle
   -k <int>      number of elements of local tensor
   --latestlink  working directory
   --cmaddr      IP address of a WSE
   --roi_px      starting column index of region of interest
   --roi_py      starting row index of region of interest
   --roi_w       width of region of interest
   --roi_h       height of region of interest
"""


import argparse
import os


def parse_args():
    """command parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", default=1, type=int, help="number of rows")
    parser.add_argument("-n", default=1, type=int, help="number of columns")
    parser.add_argument("-k", default=1, type=int, help="size of local tensor")
    parser.add_argument(
        "--latestlink", help="folder to contain the log files (default: latest)"
    )
    parser.add_argument(
        "--cmaddr", help="CM address and port, i.e. <IP>:<port>"
    )
    parser.add_argument(
        "--arch", help="wse2 or wse3. Default is wse2 when not supplied."
    )
    parser.add_argument(
        "--channels", default=1, type=int, help="number of channels"
    )
    parser.add_argument(
        "--roi_px", default=1, type=int, help="starting column index of ROI"
    )
    parser.add_argument(
        "--roi_py", default=1, type=int, help="starting row index of ROI"
    )
    parser.add_argument("--roi_w", default=3, type=int, help="width of ROI")
    parser.add_argument("--roi_h", default=3, type=int, help="height of ROI")
    parser.add_argument(
        "--use_col_major",
        action="store_true",
        help="use column major to send the row or column broadcast",
    )
    parser.add_argument(
        "--is_row_bcast",
        action="store_true",
        help="row broadcast or column broadcast",
    )
    parser.add_argument("--fabric-dims", help="Fabric dimension, i.e. <W>,<H>")
    parser.add_argument(
        "--loop_count",
        default=1,
        type=int,
        help="number of back-to-back H2D/D2H",
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

    return args, logs_dir
