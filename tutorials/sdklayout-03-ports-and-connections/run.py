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

#################################
### Sender 1 and port 'tx1_port'
#################################
sender1 = layout.create_code_region('./sender.csl', 'sender1', 1, 1)
# Color 'tx1' is scoped because even though the name of the color is
# 'tx' for both senders, colors must be globally unique for the
# compiler to assign different values to them. By scoping colors like
# this we are effectively uniqueing them since code regions are unique
# (i.e., no two regions can have the same name).
tx1 = sender1.color('tx')
sender1.set_param_all('size', size)
sender1.set_param_all(tx1)
# A sender port is created using a color ('tx1'), an edge (in this
# example the edge doesn't matter since we have a 1x1 code region),
# a list of routing positions and a size. The routing positions for an
# output port must not contain output routes (if they do, an error will
# be emitted). That's because the compiler is free to chose any output
# route depending on what's globally optimal. Finally, the 'size' is
# used to verify compatibility between connected ports.
tx1_port = sender1.create_output_port(tx1, Edge.RIGHT, [sender_routes], size)


#################################
### Sender 2 and port 'tx2_port'
#################################
sender2 = layout.create_code_region('./sender.csl', 'sender2', 1, 1)
tx2 = sender2.color('tx')
sender2.set_param_all('size', size)
sender2.set_param_all(tx2)
tx2_port = sender2.create_output_port(tx2, Edge.RIGHT, [sender_routes], size)

#########################
### Placement of senders
#########################
# We place the senders in arbitrary locations in the layout as
# an example that demonstrates the ability of the framework to automatically
# produce paths between input and output ports.
sender1.place(2, 2)
sender2.place(4, 7)

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

#############
### Receiver
#############
receiver = layout.create_code_region('./receiver.csl', 'receiver', 1, 1)
rx = Color('rx')
receiver.set_param_all('size', size)
receiver.set_param_all(rx)
rx_port = receiver.create_input_port(rx, Edge.LEFT, [receiver_routes], size,)
receiver.place(3, 3)

#####################
### Port connections
#####################
# This is the key part of this example. The ports defined above for
# each code region, are now connected. The physical location of the
# ports can be arbitrary because the SdkLayout compiler will find
# optimal paths automatically.
layout.connect(tx1_port, rx1_port)
layout.connect(tx2_port, rx2_port)
layout.connect(tx_port, rx_port)

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
runtime.stop()

#################
### Verification
#################
# Finally, once execution has stopped, read the result from the receiver's
# memory and compare with expected value.
expected = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20], dtype=np.uint16)
actual = runtime.read_symbol(3, 3, 'data').view(np.uint16)
assert np.array_equal(expected, actual)
print("SUCCESS!")
