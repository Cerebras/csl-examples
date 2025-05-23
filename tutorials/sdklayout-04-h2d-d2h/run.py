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

from cerebras.sdk.runtime.sdkruntimepybind import (
    Color,
    Edge,
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

######################
### Common invariants
######################
size = 10
sender_routes = RoutingPosition().set_input([Route.RAMP])
receiver_routes = RoutingPosition().set_output([Route.RAMP])

############
### Add2vec
############
add2vec = layout.create_code_region('./add2vec.csl', 'add2vec', 1, 1)
rx1 = Color('rx1')
rx2 = Color('rx2')
tx = Color('tx')
add2vec.set_param_all('size', size)
add2vec.set_param_all(rx1)
add2vec.set_param_all(rx2)
add2vec.set_param_all(tx)
rx1_port = add2vec.create_input_port(rx1, Edge.RIGHT, [receiver_routes], size,)
rx2_port = add2vec.create_input_port(rx2, Edge.RIGHT, [receiver_routes], size,)
tx_port = add2vec.create_output_port(tx, Edge.LEFT, [sender_routes], size,)
add2vec.place(7, 4)

#############################
### Input and output streams
#############################
in_stream1 = layout.create_input_stream(rx1_port)
in_stream2 = layout.create_input_stream(rx2_port)
out_stream = layout.create_output_stream(tx_port)

#################
### Compilation
#################
# Compile the layout and use 'out' as the prefix for all
# produced artifacts.
compile_artifacts = layout.compile(out_prefix='out')

#############
### Runtime
#############
# Create the runtime using the compilation artifacts and the execution platform.
runtime = SdkRuntime(compile_artifacts, platform, memcpy_required=False)
runtime.load()
runtime.run()

# Data to be sent to the device and buffer to accept the result from
# the device.
data1 = np.random.randint(0, 1000, size=size, dtype=np.uint32)
data2 = np.random.randint(1000, 2000, size=size, dtype=np.uint32)
actual = np.empty(size, dtype=np.uint32)

# Send and receive to/from the device.
runtime.send(in_stream1, data1, nonblock=True)
runtime.send(in_stream2, data2, nonblock=True)
runtime.receive(out_stream, actual, size, nonblock=True)

runtime.stop()

#################
### Verification
#################
# Verify that the received data is correct.
expected = data1 + data2
assert np.array_equal(expected, actual)
print("SUCCESS!")
