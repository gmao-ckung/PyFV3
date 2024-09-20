from gt4py.cartesian.gtscript import PARALLEL, FORWARD, BACKWARD, computation, interval

from ndsl import StencilFactory, QuantityFactory
from ndsl.dsl.typing import FloatField, FloatFieldIJ, Float, BoolFieldIJ
from ndsl.stencils.testing import TranslateFortranData2Py

def W_fix_consrv_moment(
    w: FloatField,
    w2: FloatField,
    dp2: FloatField,
    gz: FloatFieldIJ,
    w_max: Float,
    w_min: Float,
    compute_performed: BoolFieldIJ,
):
    """
    Args:
        w (in/out):
        w2 (in?):
        dp2(in):
        w_max(in):
        w_min(in):
        compute_performed: (Internal Temporary),
    """

    with computation(PARALLEL), interval(...):
        w2 = w

    with computation(FORWARD): 
        with interval(0,1):
            compute_performed = False
            if(w2 > w_max):
                gz = (w2 - w_max) * dp2
                w2 = w_max
                compute_performed = True
            elif(w2 < w_min):
                gz = (w2 - w_min) * dp2
                w2 = w_min
                compute_performed = True
        with interval(1,-1):
            if(compute_performed):
                w2 = w2 + gz / dp2
                compute_performed = False
            if(w2 > w_max):
                gz = (w2 - w_max) * dp2
                w2 = w_max
                compute_performed = True
            elif(w2 < w_min):
                gz = (w2 - w_min) * dp2
                w2 = w_min
                compute_performed = True

    with computation(BACKWARD):
        with interval(-1,None):
            if(w2 > w_max):
                gz = (w2 - w_max) * dp2
                w2 = w_max
            elif(w2 < w_min):
                gz = (w2 - w_min) * dp2
                w2 = w_min
        with interval(1,-1):
            w2 = w2 + gz / dp2
            if(w2 > w_max):
                gz = (w2 - w_max) * dp2
                w2 = w_max
            elif(w2 < w_min):
                gz = (w2 - w_min) * dp2
                w2 = w_min

class testClass:
    """
    Class to test with DaCe orchestration. test class is MoistCVPlusPt_2d
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        grid,
    ):
        

        self._w_fix_consrv_moment = stencil_factory.from_origin_domain(
            func=W_fix_consrv_moment,
            origin=(3,3,0),
            domain=(24,1,72),
        )

    def __call__(
        self,
        w: FloatField,
        w2: FloatField,
        dp2: FloatField,
        gz: FloatFieldIJ,
        w_max: Float,
        w_min: Float,
    ):
        self._w_fix_consrv_moment(
            w,
            w2,
            dp2,
            gz,
            w_max,
            w_min,
        )


class TranslateW_fix_consrv_moment(TranslateFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, stencil_factory)
        self.stencil_factory = stencil_factory
        self.grid = grid
        self.compute_func = testClass(self.stencil_factory, self.grid)  # type: ignore
        self.quantity_factory = grid.quantity_factory

        print("Running W_fix_consrv_moment translate test")
        print("npz = ", grid.npz)
        self.in_vars["data_vars"] = {
            "w": {
                "kend": grid.npz-1,
                },
            "w2": {"istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz-1,},
            "dp2_W": {"istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz-1,},

        }

        self.in_vars["parameters"] = ["w_max", "w_min"]

        self.out_vars = {
            "w": {
                "kend": grid.npz-1,
                },
            "w2": {"istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz-1,},
            "dp2_W": {"istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz-1,},
        }
        self._gz = self.quantity_factory._numpy.zeros(
            (
                31,
                31,
            ), dtype=Float,
        )

        self._compute_performed = self.quantity_factory._numpy.zeros(
            (
                31,
                31,
            ), dtype=bool,
        )

    def compute_from_storage(self, inputs):

        # print("inputs[w] shape = ", inputs["w"].shape)
        # print("inputs[w2] shape = ", inputs["w2"].shape)
        # print('self.grid.is_ = ', self.grid.is_)
        # print('self.grid.ie = ', self.grid.ie)
        # print('self.grid.js = ', self.grid.js)
        # print('self.grid.je = ', self.grid.je)
        # print('self.storage_vars() = ', self.storage_vars())
        # self.make_storage_data_input_vars(inputs)
        # exit(1)
        self.compute_func(
                     inputs["w"],
                     inputs["w2"],
                     inputs["dp2_W"],
                     self._gz,
                     inputs["w_max"],
                     inputs["w_min"],
                     self._compute_performed
                     )
        
        for i in range(3,27):
            for k in range(72):
                temp = inputs["w2"][i,3,k]
                inputs["w2"][i,:,k] = temp
        return inputs
