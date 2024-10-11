from ndsl import StencilFactory, QuantityFactory
from ndsl.dsl.typing import FloatField, FloatFieldIJ, Float, BoolFieldIJ, IntField, IntFieldIJ, Float
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.stencils.testing import TranslateFortranData2Py
from ndsl.stencils.testing.grid import Grid
from gt4py.cartesian.gtscript import PARALLEL, FORWARD, BACKWARD, computation, interval
from pyFV3.stencils.map_single import lagrangian_contributions

class test_Lagragian_Contribution:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        grid: Grid,
    ):
        print("In test_Lagragian_Contribution")

        self._lagrangian_contributions = stencil_factory.from_origin_domain(
            func=lagrangian_contributions,
            origin=(3,3,0),
            domain=(24,1,72),
        )

    def __call__(
        self,
        km: int,
        not_exit_loop: BoolFieldIJ,
        INDEX_LM1: IntField,
        INDEX_LP0: IntField,
        # q2: FloatField,
        q: FloatField,
        pe1: FloatField,
        pe2: FloatField,
        q4_1: FloatField,
        q4_2: FloatField,
        q4_3: FloatField,
        q4_4: FloatField,
        dp1: FloatField,
        lev: IntFieldIJ,
        K0: IntField,
    ):
        self._lagrangian_contributions(
            km,
            not_exit_loop,
            INDEX_LM1,
            INDEX_LP0,
            # q2,
            q,
            pe1,
            pe2,
            q4_1,
            q4_2,
            q4_3,
            q4_4,
            dp1,
            lev,
            K0,
        )

class TranslateLagrangian_Contribution(TranslateFortranData2Py):
    def __init__(self, grid: Grid, namelist, stencil_factory):
        super().__init__(grid, stencil_factory)
        self.stencil_factory = stencil_factory
        self.grid = grid
        self.compute_func = test_Lagragian_Contribution(self.stencil_factory, self.grid)  # type: ignore
        self.quantity_factory = grid.quantity_factory

        self.in_vars["data_vars"] = {
            "q1": {
                "kend": grid.npz-1,
                },

            "pe1_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz
            },
            "pe2_": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz
            },
            "q4_1": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz-1,
                },
            "q4_2": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz-1,
            },
            "q4_3": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz-1,
            },
            "q4_4": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz-1,
            },
            "dp1_":{
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz-1,
            }
        }

        self.out_vars = {
            "q1": {
                "kend": grid.npz-1,
                },
            
            # "q2": {
            #     "kend": grid.npz-1,
            #     },

            # "pe1_": {
            #     "istart": grid.is_,
            #     "iend": grid.ie,
            #     "jstart": grid.js,
            #     "jend": grid.je,
            #     "kend": grid.npz
            # },
            # "pe2_": {
            #     "istart": grid.is_,
            #     "iend": grid.ie,
            #     "jstart": grid.js,
            #     "jend": grid.je,
            #     "kend": grid.npz
            # },
            # "q4_1": {
            #     "istart": grid.is_,
            #     "iend": grid.ie,
            #     "jstart": grid.js,
            #     "jend": grid.je,
            #     "kend": grid.npz-1,
            #     },
            # "q4_2": {
            #     "istart": grid.is_,
            #     "iend": grid.ie,
            #     "jstart": grid.js,
            #     "jend": grid.je,
            #     "kend": grid.npz-1,
            # },
            # "q4_3": {
            #     "istart": grid.is_,
            #     "iend": grid.ie,
            #     "jstart": grid.js,
            #     "jend": grid.je,
            #     "kend": grid.npz-1,
            # },
            # "q4_4": {
            #     "istart": grid.is_,
            #     "iend": grid.ie,
            #     "jstart": grid.js,
            #     "jend": grid.je,
            #     "kend": grid.npz-1,
            # },
            # "dp1_":{
            #     "istart": grid.is_,
            #     "iend": grid.ie,
            #     "jstart": grid.js,
            #     "jend": grid.je,
            #     "kend": grid.npz-1,
            # }
        }

    def compute_from_storage(self, inputs):

        # print('self._not_exit_loop shape: ',self._not_exit_loop.shape)
        # print('self._INDEX_LM1 shape: ', self._INDEX_LM1.shape)
        # print('self._INDEX_LP0 shape: ', self._INDEX_LP0.shape))
        # print('inputs["q1"] shape: ', inputs["q1"].shape)
        # print('inputs["pe1"] shape:',inputs["pe1"].shape)
        # print('inputs["pe2"] shape:',inputs["pe2"].shape)
        # print('inputs["q4_1"] shape:',inputs["q4_1"].shape)
        # print('inputs["q4_2"] shape:',inputs["q4_2"].shape)
        # print('inputs["q4_3"] shape:',inputs["q4_3"].shape)
        # print('inputs["q4_4"] shape:',inputs["q4_4"].shape)
        # print('inputs["dp1_"] shape:',inputs["dp1_"].shape)
        # print('self._lev shape:', self._lev.shape)

        self._not_exit_loop = self.quantity_factory.zeros(
            [X_DIM, Y_DIM],
            units="",
            dtype=bool,
        )

        self._INDEX_LM1 = self.quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="",
            dtype=int,
        )

        self._INDEX_LP0 = self.quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="",
            dtype=int,
        )

        self._lev = self.quantity_factory.zeros(
            [X_DIM, Y_DIM],
            units="",
            dtype=int,
        )

        self._K0 = self.quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="",
            dtype=Float,
        )

        self.compute_func(
                    self.grid.npz,
                    self._not_exit_loop,
                    self._INDEX_LM1,
                    self._INDEX_LP0,
                    inputs["q1"],
                    inputs["pe1_"],
                    inputs["pe2_"],
                    inputs["q4_1"],
                    inputs["q4_2"],
                    inputs["q4_3"],
                    inputs["q4_4"],
                    inputs["dp1_"],
                    self._lev,
                    self._K0
                )

        # print("k0[3,3,:] = ", self._K0.data[3,3,:])

        return inputs