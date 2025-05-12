import m5
import argparse

from m5.objects import System, SrcClockDomain, VoltageDomain, Root
from m5.objects import RiscvO3CPU, Cache, AddrRange, SEWorkload, Process
from m5.objects import MemCtrl, DDR4_2400_8x8, GDDR5_4000_2x32 , DDR3_1600_8x8, SystemXBar, L2XBar
from m5.objects import RiscvISA
from m5.objects import *
from m5.objects.Prefetcher import *


parser = argparse.ArgumentParser()

parser.add_argument(
    "--bin",
    default="",
    type=str
)
parser.add_argument(
    "--vlen",
    default=256,
    type=int
)
parser.add_argument(
    "--crosslat",
    default=6,
    type=int
)
args = parser.parse_args()


bin = args.bin
vlen = args.vlen
crosslat = args.crosslat


class L1Cache(Cache):
    assoc = 4
    tag_latency = 2
    data_latency = 2
    response_latency = 2
    mshrs = 4
    tgts_per_mshr = 20

    def connectCPU(self, cpu):
        raise NotImplementedError
    
    def connectBus(self, bus):
        self.mem_side = bus.cpu_side_ports

class L1ICache(L1Cache):
    size = '32kB'

    def connectCPU(self, cpu):
        self.cpu_side = cpu.icache_port

class L1DCache(L1Cache):
    size = '32kB'

    def connectCPU(self, cpu):
        self.cpu_side = cpu.dcache_port

class SmsPrefetcherMy(SmsPrefetcher):
    ft_size = Param.Unsigned(32, "Size of Filter and Active generation table")
    pht_size = Param.Unsigned(16384, "Size of pattern history table")
    region_size = Param.Unsigned(4096, "Spatial region size")

    queue_squash = True
    queue_filter = True
    cache_snoop = True
    prefetch_on_access = True
    on_inst = True

class L2Cache(Cache):
    size = '512kB'
    assoc = 2
    tag_latency  = 40
    data_latency = 40
    response_latency = 40
    mshrs = 40
    tgts_per_mshr = 12
    prefetcher=SmsPrefetcherMy()

    def connectCPUSideBus(self, bus):
        self.cpu_side = bus.mem_side_ports

    def connectMemSideBus(self, bus):
        self.mem_side = bus.cpu_side_ports


system = System()
system.clk_domain = SrcClockDomain()
system.clk_domain.clock = '1.6GHz'
system.clk_domain.voltage_domain = VoltageDomain()

system.mem_mode = 'timing'
system.mem_ranges = [AddrRange('8192MB')]

system.cpu = RiscvO3CPU()

# Create the L1 caches
system.cpu.icache = L1ICache()
system.cpu.dcache = L1DCache()

# Connect the caches to the CPU
system.cpu.icache.connectCPU(system.cpu)
system.cpu.dcache.connectCPU(system.cpu)

system.l2bus = L2XBar()

system.cpu.icache.connectBus(system.l2bus)
system.cpu.dcache.connectBus(system.l2bus)

system.l2cache = L2Cache()
system.l2cache.connectCPUSideBus(system.l2bus)

system.membus = SystemXBar()
system.l2cache.connectMemSideBus(system.membus)

system.cpu.createInterruptController()

system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR4_2400_8x8()
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports

system.cpu.isa = RiscvISA(vlen=vlen)

system.workload = SEWorkload.init_compatible(bin)

process = Process()
process.cmd = [bin]
system.cpu.workload = process
system.multi_thread = True
system.cpu.createThreads()

system.cpu.fetchWidth          = 6
system.cpu.decodeWidth         = 2
system.cpu.issueWidth          = 4
system.cpu.renameWidth         = 4
system.cpu.commitWidth         = 6
system.cpu.commitToRenameDelay = 1
system.cpu.numIQEntries        = 24#60
system.cpu.wbWidth             = 8
system.cpu.dispatchWidth       = 8

class Int_ALU(FUDesc):
    opList = [
        OpDesc(opClass="IntAlu", opLat=1, pipelined=True)
    ]
    count = 2

class Int_Mult_Unit(FUDesc):
    opList = [
        OpDesc(opClass="IntMult", opLat=3, pipelined=True),
    ]
    count = 2

class Int_Div_Unit(FUDesc):
    opList = [
        OpDesc(opClass="IntDiv", opLat=20, pipelined=False),
    ]
    count = 1

class FP_ALU_Unit(FUDesc):
    opList = [
        OpDesc(opClass="FloatAdd",  opLat=4, pipelined=True),
        OpDesc(opClass="FloatCmp",  opLat=4, pipelined=True),
        OpDesc(opClass="FloatCvt",  opLat=4, pipelined=True),
        OpDesc(opClass="FloatMisc", opLat=4, pipelined=True),
    ]
    count = 2

class FP_Mult_Unit(FUDesc):
    opList = [
        OpDesc(opClass="FloatMult",    opLat=4, pipelined=True),
        OpDesc(opClass="FloatMultAcc", opLat=5, pipelined=True)
    ]
    count = 2

class FP_Div_Unit(FUDesc):
    opList = [
        OpDesc(opClass="FloatDiv",  opLat=28, pipelined=False),
    ]
    count = 1

class FP_Sqrt_Unit(FUDesc):
    opList = [
        OpDesc(opClass="FloatSqrt", opLat=21, pipelined=False),
    ]
    count = 1

class LSU(FUDesc):
    opList = [
        OpDesc(opClass="MemWrite",      opLat=3, pipelined=True),
        OpDesc(opClass="MemRead",       opLat=1, pipelined=True),
        OpDesc(opClass="FloatMemWrite", opLat=3, pipelined=True),
        OpDesc(opClass="FloatMemRead",  opLat=1, pipelined=True),
    ]
    count = 2

class RVV_Int_ALU(FUDesc):
    opList = [
        OpDesc(opClass="SimdConfig",    opLat=3, pipelined=True),
        OpDesc(opClass="SimdAlu",       opLat=1, pipelined=True),
        OpDesc(opClass="SimdAdd",       opLat=1, pipelined=True),
        OpDesc(opClass="SimdCvt",       opLat=3, pipelined=True),
        OpDesc(opClass="SimdReduceAlu", opLat=5, pipelined=True),
        OpDesc(opClass="SimdReduceCmp", opLat=5, pipelined=True),
        OpDesc(opClass="SimdReduceAdd", opLat=5, pipelined=True),
        OpDesc(opClass="SimdShift",     opLat=3, pipelined=True),
    ]
    count = 2

class RVV_Int_Mult_Unit(FUDesc):
    opList = [
        OpDesc(opClass="SimdMult",    opLat=4, pipelined=True),
        OpDesc(opClass="SimdMultAcc", opLat=4, pipelined=True),
    ]
    count = 2

class RVV_Int_Div_Unit(FUDesc):
    opList = [
        OpDesc(opClass="SimdDiv", opLat=21, pipelined=False)
    ]
    count = 1

class RVV_Misc_Unit(FUDesc):
    opList = [
        OpDesc(opClass="SimdExt",  opLat=4, pipelined=True),
        OpDesc(opClass="SimdCmp",  opLat=4, pipelined=True),
        OpDesc(opClass="SimdMisc", opLat=12, pipelined=True),
    ]
    count = 1

class RVV_FP_ALU(FUDesc):
    opList = [
        OpDesc(opClass="SimdFloatAlu",       opLat=5, pipelined=True),
        OpDesc(opClass="SimdFloatAdd",       opLat=5, pipelined=True),
        OpDesc(opClass="SimdFloatCmp",       opLat=5, pipelined=True),
        OpDesc(opClass="SimdFloatExt",       opLat=5, pipelined=True),
        OpDesc(opClass="SimdFloatMisc",      opLat=5, pipelined=True),
        OpDesc(opClass="SimdFloatCvt",       opLat=6, pipelined=True),
        OpDesc(opClass="SimdFloatReduceAdd", opLat=7,  pipelined=True),
        OpDesc(opClass="SimdFloatReduceCmp", opLat=7, pipelined=True)
    ]
    count = 2

class RVV_FP_Mult_Unit(FUDesc):
    opList = [
        OpDesc(opClass="SimdFloatMult",    opLat=6, pipelined=True),
        OpDesc(opClass="SimdFloatMultAcc", opLat=6, pipelined=True),
        OpDesc(opClass="SimdFloatMatMultAcc"),
        OpDesc(opClass="VectorFloatVecMathArith", opLat=crosslat, pipelined=True)
    ]
    count = 2

class RVV_FP_Div_Unit(FUDesc):
    opList = [
        OpDesc(opClass="SimdFloatDiv" , opLat=28, pipelined=False),
    ]
    count = 1

class RVV_FP_Sqrt_Unit(FUDesc):
    opList = [
        OpDesc(opClass="SimdFloatSqrt", opLat=21, pipelined=False),
    ]
    count = 1

class RVV_LSU(FUDesc):
    opList = [
        OpDesc(opClass="SimdWholeRegisterLoad"           , opLat=4, pipelined=True),
        OpDesc(opClass="SimdUnitStrideLoad"              , opLat=5, pipelined=True),
        OpDesc(opClass="SimdStridedLoad"                 , opLat=7, pipelined=True),
        OpDesc(opClass="SimdIndexedLoad"                 , opLat=7, pipelined=True),
        OpDesc(opClass="SimdUnitStrideMaskLoad"          , opLat=5, pipelined=True),
        OpDesc(opClass="SimdUnitStrideSegmentedLoad"     , opLat=7, pipelined=True),
        OpDesc(opClass="SimdUnitStrideFaultOnlyFirstLoad", opLat=10, pipelined=True),

        OpDesc(opClass="SimdWholeRegisterStore"      , opLat=3, pipelined=True),
        OpDesc(opClass="SimdUnitStrideStore"         , opLat=5, pipelined=True),
        OpDesc(opClass="SimdStridedStore"            , opLat=7, pipelined=True),
        OpDesc(opClass="SimdIndexedStore"            , opLat=7, pipelined=True),
        OpDesc(opClass="SimdUnitStrideMaskStore"     , opLat=5, pipelined=True),
        OpDesc(opClass="SimdUnitStrideSegmentedStore", opLat=7, pipelined=True),
        
    ]
    count = 2

class CPU_FUPool(FUPool):
    FUList = [
        # Scalar
        Int_ALU(),
        Int_Mult_Unit(),
        Int_Div_Unit(),
        FP_ALU_Unit(),
        FP_Mult_Unit(),
        FP_Div_Unit(),
        FP_Sqrt_Unit(),
        LSU(),
        # RVV
        RVV_Int_ALU(),
        RVV_Int_Mult_Unit(),
        RVV_Int_Div_Unit(),
        RVV_Misc_Unit(),
        RVV_FP_ALU(),
        RVV_FP_Mult_Unit(),
        RVV_FP_Div_Unit(),
        RVV_FP_Sqrt_Unit(),
        RVV_LSU()
    ]

system.cpu.fuPool = CPU_FUPool()

# Run SE mode
root = Root(full_system=False, system=system)
m5.instantiate()
print("Running " + bin)
print("Beginning simulation!")
exit_event = m5.simulate()
print("Exiting")
