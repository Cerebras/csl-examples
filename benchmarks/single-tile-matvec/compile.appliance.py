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
import json
from google.protobuf.json_format import MessageToJson

#from tests.appliance.common.cluster_details_utils import build_cluster_details
#import cluster_details_utils

from cerebras.sdk.client import (
        SdkCompiler,
)

hash_filename = "hash.json"

compiler = SdkCompiler()

hashstr = compiler.compile("./src", "layout_matvec.csl", "--arch wse3 --fabric-dims=9,4 --fabric-offsets=4,1 --params=width:2,height:2,tile_size:25,iters:1 -o latest --memcpy --channels=1")

print("compile artifact:", hashstr)

print(f"dump artifact name to file {hash_filename}")
with open(hash_filename, "w") as write_file:
    json.dump(hashstr, write_file)

