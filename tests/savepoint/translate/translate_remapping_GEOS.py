import ndsl.dsl.gt4py_utils as utils
from ndsl import Namelist, StencilFactory
from pyFV3 import DynamicalCoreConfig
from pyFV3.stencils.remapping import init_pe, moist_cv_pt_pressure
from pyFV3.stencils import moist_cv
from ndsl.stencils.testing import pad_field_in_j
from pyFV3.testing import TranslateDycoreFortranData2Py
from ndsl.constants import (
    X_DIM,
    X_INTERFACE_DIM,
    Y_DIM,
    Y_INTERFACE_DIM,
    Z_DIM,
    Z_INTERFACE_DIM,
)
from ndsl.dsl.typing import Float

class TranslateRemapping_GEOS(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        
        self.in_vars["data_vars"] = {
            
            "pe_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz + 1,
                # "kaxis": 1,
            },
            "pe1_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz + 1,
                # "kaxis": 1,
            },
            "pe2_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz + 1,
                # "kaxis": 1,
            },
            # Note that tracers are i x k shaped.
            # Setting "axis" as 1 enables the translate test to read them properly
            "qvapor": {
                "axis": 1
            },
            "qliquid": {
                "axis": 1
            },
            "qice": {
                "axis": 1
            },
            "qrain": {
                "axis": 1
            },
            "qsnow": {
                "axis": 1
            },
            "qgraupel": {
                "axis": 1
            },
            "delp": {},
            "delz": {},
            "q_con": {},
            "pt": {},
            "cappa": {},
        }
        self.in_vars["parameters"] = [
            "ptop",
            "r_vir",
            # "remap_t", # For some reason, translate test can't accept a logical variable
            # "akap",
            # "zvir",
            # "last_step",
            # "consv_te",
            # "mdt",
            # "nq",
        ]
        self.out_vars = {
            "pe1_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz + 1,
                # "kaxis": 1,
            },
            "pe2_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz + 1,
                # "kaxis": 1,
            },
            # "delp": {},
            # "delz": {},
            # "q_con": {},
            # "pt": {},
            "cappa": {},
        }

        self.stencil_factory = stencil_factory
        # self.namelist found in TranslateDycoreFortranData2Py
        # self.namelist = DynamicalCoreConfig.from_namelist(namelist)
        config=DynamicalCoreConfig.from_namelist(self.namelist).remapping

        hydrostatic = config.hydrostatic
        if hydrostatic:
            raise NotImplementedError("Hydrostatic is not implemented")

        grid_indexing = stencil_factory.grid_indexing
        self._domain_jextra = (
            grid_indexing.domain[0],
            grid_indexing.domain[1] + 1,
            grid_indexing.domain[2] + 1,
        )
        
        self._init_pe = stencil_factory.from_origin_domain(
            init_pe, 
            # origin=(3,3,0),
            origin=grid_indexing.origin_compute(), 
            domain=(24,1,73),
        )

        self._moist_cv_pt_pressure = stencil_factory.from_origin_domain(
            moist_cv_pt_pressure,
            # externals={"kord_tm": config.kord_tm, "hydrostatic": hydrostatic},
            externals={"hydrostatic": hydrostatic},
            origin=grid_indexing.origin_compute(),
            # domain=grid_indexing.domain_compute(add=(0, 0, 1)),
            domain=(grid.nic, 1, grid.npz+1), # Note : Many intervals go from (0,-1) in this stencil
        )

    def compute_from_storage(self, inputs):       
        
        # print("inputs[qvapor].data.shape() 1 = ", inputs["qvapor"].data.shape)

        # Replicates tracer values in I along the J direction
        for name, value in inputs.items():
            if hasattr(value, "shape") and len(value.shape) > 1 and value.shape[1] == 1:
                inputs[name] = self.make_storage_data(
                    pad_field_in_j(
                        value, self.grid.njd, backend=self.stencil_factory.backend
                    )
                )
        # print("inputs[qvapor].data.shape() 2 = ", inputs["qvapor"].data.shape)
        self._init_pe(
            inputs["pe_"],
            inputs["pe1_"],
            inputs["pe2_"],
            inputs["ptop"],
        )

        self._moist_cv_pt_pressure(
            inputs["qvapor"],
            inputs["qliquid"],
            inputs["qrain"],
            inputs["qsnow"],
            inputs["qice"],
            inputs["qgraupel"],
            inputs["q_con"],
            inputs["pt"],
            inputs["cappa"],
            inputs["delp"],
            inputs["delz"],
            True,
            Float(inputs["r_vir"]),
        )
        return inputs
