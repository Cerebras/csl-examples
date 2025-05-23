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

import numpy as np

from cerebras.geometry.geometry import IntVector
from cerebras.sdk.runtime.sdkruntimepybind import (
    Color,
    Route,
    RoutingPosition,
    SdkLayout,
    SdkTarget,
    SdkRuntime,
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

###########
### Layout
###########
# If 'cmaddr' is empty then we create a default simulation layout.
# If 'cmaddr' is not empty then 'config' and 'target' are ignored.
config = SimfabConfig(dump_core=True)
target = SdkTarget.WSE3 if (args.arch == 'wse3') else SdkTarget.WSE2
platform = get_platform(args.cmaddr, config, target)
layout = SdkLayout(platform)

################
### Code region
################
# Create a 2x1 code region using 'send_receive.csl' as the source code.
# The code region is given the name 'send_receive'. The first PE will
# act as the sender and the second PE as the receiver.
code = layout.create_code_region('./send_receive.csl', 'send_receive', 2, 1)
# The 'set_param' method will set the value of a parameter on a specific
# PE using its local coordinates, i.e., the coordinates with respect
# to the respective code region and not the global coordinates.
# Here we set 'select=0' for the sender (coordinates (0, 0)) and
# 'select=1' for the receiver (coordinates (1, 0)).
# PE coordinates can be created using the 'IntVector' class that represents
# 2D grid coordinates.
sender_coords = IntVector(0, 0)
receiver_coords = IntVector(1, 0)
code.set_param(sender_coords, 'select', 0)
code.set_param(receiver_coords, 'select', 1)

#########################################
### Routing between sender and receiver
#########################################
# The sender routes traffic from the RAMP to the EAST where the
# receiver will receive the data (from the WEST) and route them to the RAMP.
send_routes = RoutingPosition().set_input([Route.RAMP]).set_output([Route.EAST])
receive_routes = RoutingPosition().set_input([Route.WEST]).set_output([Route.RAMP])
# Define a symbolic color. The SdkLayout compiler will resolve this into
# a physical value.
c = Color('c')
# Use color 'c' to paint the routes from sender to receiver.
code.paint(sender_coords, c, [send_routes])
code.paint(receiver_coords, c, [receive_routes])
code.set_param_all(c)

#################
### Compilation
#################
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
# Finally, once execution has stopped, read the result from the receiver's
# memory and compare with expected value.
expected = np.array([1, 2, 3, 4, 5], dtype=np.uint16)
# The 'read_symbol' method will read a symbol from memory at the specified
# global coordinates and return it as a numpy array of type 'dtype'.
actual = runtime.read_symbol(1, 0, 'buffer', dtype='uint16')
assert np.array_equal(expected, actual)
print("SUCCESS!")
