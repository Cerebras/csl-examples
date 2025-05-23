#!/usr/bin/env cs_python

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

from cerebras.sdk.runtime.sdkruntimepybind import (
    SdkRuntime,
    SdkTarget,
    SdkLayout,
    SimfabConfig,
    get_platform,
)

parser = argparse.ArgumentParser()
parser.add_argument('--cmaddr', help='IP:port for CS system')
parser.add_argument(
    '--arch',
    choices=['wse2', 'wse3'],
    default='wse3',
    help='Target WSE architecture (default: wse3)'
)
args = parser.parse_args()

########################################
### Layout, code region and compilation
########################################
value = 550
# If 'cmaddr' is empty then we create a default simulation layout.
# If 'cmaddr' is not empty then 'config' and 'target' are ignored.
config = SimfabConfig(dump_core=True)
target = SdkTarget.WSE3 if (args.arch == 'wse3') else SdkTarget.WSE2
platform = get_platform(args.cmaddr, config, target)
layout = SdkLayout(platform)
# Create a 1x1 code region using 'gv.csl' as the source code.
# The code region is called 'gv'.
code = layout.create_code_region('./gv.csl', 'gv', 1, 1)
# Set the 'value' param on all PEs in the region. In this
# example 'code' has only one PE.
code.set_param_all('value', value)
# Compile the layout and use 'out' as the prefix for all
# produced artifacts.
compile_artifacts = layout.compile(out_prefix='out')

############
### Runtime
############
# Create the runtime using the compilation artifacts and the execution platform.
runtime = SdkRuntime(compile_artifacts, platform, memcpy_required=False)
runtime.load()
runtime.run()
runtime.stop()

#################
### Verification
#################
# Finally, once execution has stopped, read 'value' from device memory
# and compare it against the expected value.
result = runtime.read_symbol(0, 0, 'gv', dtype='uint16')
assert result == [value]
print("SUCCESS!")
