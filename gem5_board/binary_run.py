
import sys

import m5
from m5.objects import Root

from gem5.utils.requires import requires
from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.memory import DualChannelDDR4_2400
from gem5.components.processors.simple_processor import (
    SimpleProcessor,
)
from gem5.components.processors.cpu_types import CPUTypes
from gem5.isas import ISA
from gem5.coherence_protocol import CoherenceProtocol
from gem5.resources.resource import CustomResource
from gem5.simulate.simulator import Simulator

from CoreRVV import RVVProcessor

# This runs a check to ensure the gem5 binary is compiled for RISCV.

requires(
    isa_required=ISA.RISCV,
)

# With RISCV, we use simple caches.
from gem5.components.cachehierarchies.classic\
    .private_l1_private_l2_cache_hierarchy import (
    PrivateL1PrivateL2CacheHierarchy,
)
# Here we setup the parameters of the l1 and l2 caches.
cache_hierarchy = PrivateL1PrivateL2CacheHierarchy(
    l1d_size="32kB",
    l1i_size="32kB",
    l2_size="1MB",
)

# Memory: Dual Channel DDR4 2400 DRAM device.
memory = DualChannelDDR4_2400(size = "8GB")

# Here we setup the processor. We use a our RVV processor.
processor = RVVProcessor(
    num_cores=1,
)

# Here we setup the board
board = SimpleBoard(
    clk_freq="3GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy,
)

# set the riscv binary as the board workload
binary_path = "binary"
if len(sys.argv)>1:
    binary_path = sys.argv[1]
board.set_se_binary_workload(CustomResource(binary_path))

# run the simulation with the RISCV Matched board
simulator = Simulator(board=board, full_system=False)
simulator.run()

print(
    "Exiting @ tick {} because {}.".format(
        simulator.get_current_tick(),
        simulator.get_last_exit_event_cause(),
    )
)