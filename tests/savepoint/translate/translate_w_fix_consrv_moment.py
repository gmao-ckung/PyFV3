from gt4py.cartesian.gtscript import PARALLEL, FORWARD, BACKWARD, computation, interval

from ndsl import StencilFactory, QuantityFactory
from ndsl.dsl.typing import FloatField, FloatFieldIJ, Float, BoolFieldIJ
from ndsl.stencils.testing.grid import Grid
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

    with computation(FORWARD), interval(0,1):
        if(w2 > (w_max * 2.0)):
            w2 = w_max * 2.0
        elif(w2 < (w_min * 2.0)):
            w2 = w_min * 2.0
    
    with computation(PARALLEL), interval(...):
        w = w2
class testClass:

    def __init__(
        self,
        stencil_factory: StencilFactory,
        grid: Grid,
    ):
        print("In testClass")
        print("Grid.nid Grid.njd Grid.npz", grid.nid, grid.njd, grid.npz)

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
        compute_performed_bool: BoolFieldIJ
    ):
        self._w_fix_consrv_moment(
            w,
            w2,
            dp2,
            gz,
            w_max,
            w_min,
            compute_performed_bool,
        )


class TranslateW_fix_consrv_moment(TranslateFortranData2Py):
    def __init__(self, grid: Grid, namelist, stencil_factory):
        super().__init__(grid, stencil_factory)
        self.stencil_factory = stencil_factory
        self.grid = grid
        self.compute_func = testClass(self.stencil_factory, self.grid)  # type: ignore
        self.quantity_factory = grid.quantity_factory

        print("Running W_fix_consrv_moment translate test")

        self.in_vars["data_vars"] = {
            "w": {
                "kend": grid.npz-1,
                },
            "w2": grid.compute_dict(),
            "dp2_W": grid.compute_dict(),

        }

        self.in_vars["parameters"] = ["w_max", "w_min"]

        self.out_vars = {
            "w": {
                "kend": grid.npz-1,
                },
            "w2": grid.compute_dict(),
            "dp2_W": grid.compute_dict(),
        }
        self._gz = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
            ), dtype=Float,
        )

        self._compute_performed = self.quantity_factory._numpy.zeros(
            (
                grid.nid,
                grid.njd,
            ), dtype=bool,
        )

    def compute_from_storage(self, inputs):

        self.compute_func(
                     inputs["w"],
                     inputs["w2"],
                     inputs["dp2_W"],
                     self._gz,
                     inputs["w_max"],
                     inputs["w_min"],
                     self._compute_performed
                     )
        
        for i in range(self.grid.is_,self.grid.ie+1):
            for k in range(self.grid.npz):
                temp = inputs["w2"][i,3,k]
                inputs["w2"][i,:,k] = temp
        return inputs
