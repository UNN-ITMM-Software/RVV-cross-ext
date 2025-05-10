
from gem5.isas import ISA
from m5.objects.RiscvCPU import RiscvO3CPU

from gem5.components.processors.base_cpu_processor import BaseCPUProcessor
from gem5.components.processors.base_cpu_core import BaseCPUCore
from gem5.components.processors.cpu_types import CPUTypes
from m5.objects.FUPool import *
from m5.objects.FuncUnit import *
from m5.objects.FuncUnitConfig import *
from m5.params import *
from m5.proxy import *

class Vector_Unit(FUDesc):
    opList = [
        OpDesc(opClass="VectorUnitStrideLoad"),
        OpDesc(opClass="VectorUnitStrideStore"),
        OpDesc(opClass="VectorUnitStrideMaskLoad"),
        OpDesc(opClass="VectorUnitStrideMaskStore"),
        OpDesc(opClass="VectorStridedLoad"),
        OpDesc(opClass="VectorStridedStore"),
        OpDesc(opClass="VectorIndexedLoad"),
        OpDesc(opClass="VectorIndexedStore"),
        OpDesc(opClass="VectorUnitStrideFaultOnlyFirstLoad"),
        OpDesc(opClass="VectorWholeRegisterLoad"),
        OpDesc(opClass="VectorWholeRegisterStore"),
        OpDesc(opClass="VectorIntegerArith", opLat=2),
        OpDesc(opClass="VectorFloatArith", opLat=5),
        OpDesc(opClass="VectorFloatConvert", opLat=5),
        OpDesc(opClass="VectorIntegerReduce", opLat=2),
        OpDesc(opClass="VectorFloatReduce", opLat=5),
        OpDesc(opClass="VectorMisc", opLat=5),
        OpDesc(opClass="VectorIntegerExtension", opLat=5),
        OpDesc(opClass="VectorConfig"),
        OpDesc(opClass="VectorFloatVecMathArith", opLat=6),
    ]
    count = 2

class RVVFUPool(FUPool):
    FUList = [
        IntALU(),
        IntMultDiv(),
        FP_ALU(),
        FP_MultDiv(),
        ReadPort(),
        SIMD_Unit(),
        PredALU(),
        WritePort(),
        RdWrPort(),
        IprPort(),
        Vector_Unit(),
    ]
    

class BaseO3CPURVV(RiscvO3CPU):
    fuPool = Param.FUPool(RVVFUPool(), "Functional Unit pool")

class CoreRVV(BaseCPUCore):
    def __init__(
        self,
        core_id,
    ):
        super().__init__(core=BaseO3CPURVV(cpu_id=core_id), isa=ISA.RISCV)
        self.core.isa[0].enable_rvv = True
        
class RVVProcessor(BaseCPUProcessor):
    def __init__(
        self, num_cores: int
    ) -> None:
        self._cpu_type = CPUTypes.O3
        super().__init__(cores=self._create_cores(num_cores))

    def _create_cores(self, num_cores):
        return [CoreRVV(core_id=i) for i in range(num_cores)]