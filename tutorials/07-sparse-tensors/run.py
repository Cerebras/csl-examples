#!/usr/bin/env cs_python

import argparse
import numpy as np
from cerebras.elf.cs_elf_runner import CSELFRunner

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='the test name')
parser.add_argument("--cmaddr", help="IP:port for CS system")
args = parser.parse_args()
name = args.name

# Path to ELF files
elf_paths = [f"{name}/bin/out_0_0.elf"]

runner = CSELFRunner(elf_paths, cmaddr=args.cmaddr)
main_color_value = 0

# Turn each tuple of two 16-bit integers into one 32-bit integer
packed = [(idx << 16) + val for idx, val in [(0, 42), (3, 26)]]
packed_tensor = np.array(packed, dtype=np.int32)

# ISL map to indicate the PE that will receive the input wavelet, along with the
# direction of the input wavelet
input_port_map = "{input[idx=0:1] -> [PE[-1,0] -> index[idx]]}"
runner.add_input_tensor(main_color_value, input_port_map, packed_tensor)

# Color along which we expect the output message
output_color = 1

# ISL map to indicate the PE that will produce the output wavelet, along with
# the direction of the output wavelet
output_port_map = "{out_tensor[idx=0:1] -> [PE[1,0] -> index[idx]]}"
runner.add_output_tensor(output_color, output_port_map, np.int16)

# Proceed with simulation; fetch the output wavelets once simulation completes
runner.connect_and_run()
result_tensor = runner.out_tensor_dict["out_tensor"]

# Ensure that the result matches our expectation
# Since zero wavelets are skipped during transmission, the `@mov16` operation
# in the code is executed only twice, once for each non-zero wavelet data
np.testing.assert_equal(result_tensor, [42, 26])
print("SUCCESS!")
