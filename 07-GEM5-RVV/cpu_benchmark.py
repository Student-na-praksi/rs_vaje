from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.private_l1_cache_hierarchy import PrivateL1CacheHierarchy
from gem5.components.memory.single_channel import SingleChannelDDR3_1600
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.isas import ISA
from gem5.resources.resource import obtain_resource
from gem5.simulate.simulator import Simulator
from gem5.resources.resource import CustomResource
from m5.objects import RiscvISA
# Import prefetcher classes
from m5.objects import (
    StridePrefetcher,
    TaggedPrefetcher,
    BOPPrefetcher,
    SignaturePathPrefetcher,
)


cache_hierarchy = PrivateL1CacheHierarchy(l1d_size="8KiB", l1i_size="8KiB")

memory = SingleChannelDDR3_1600("7GiB")

processor = SimpleProcessor(cpu_type=CPUTypes.O3, num_cores=1, isa=ISA.RISCV)
# Set VLEN on every core after the processor is constructed
for core in processor.get_cores():
    core.get_simobject().isa[0].vlen = 512   # any power of two >= 128
     



   
board = SimpleBoard(
    clk_freq="3GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy
)


# Resources can be found at
    # https://resources.gem5.org/
# x86-matrix-multiply is obtained from
    # https://resources.gem5.org/resources/x86-matrix-multiply-run?version=1.0.0



# Set the workloaad.
#binary = CustomResource("./workload/rvv_quant_matmul.bin")
#binary = CustomResource("./workload/rvv_access_patterns.bin")
#binary = CustomResource("./workload/rvv_masked.bin")
binary = CustomResource("./workload/rvv_roofline_single.bin")

board.set_se_binary_workload(binary)

simulator = Simulator(board=board)
simulator.run()