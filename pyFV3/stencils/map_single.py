from typing import Optional, Sequence

from gt4py.cartesian.gtscript import FORWARD, PARALLEL, computation, interval

from ndsl import QuantityFactory, StencilFactory, orchestrate
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ, IntField, IntFieldIJ, BoolFieldIJ # noqa: F401
from ndsl.stencils.basic_operations import copy_defn
from pyFV3.stencils.remap_profile import RemapProfile


def set_dp(dp1: FloatField, pe1: FloatField, lev: IntFieldIJ):
    with computation(PARALLEL), interval(...):
        dp1 = pe1[0, 0, 1] - pe1
    with computation(FORWARD), interval(0, 1):
        lev = 0


def lagrangian_contributions(
    km: int,
    not_exit_loop: BoolFieldIJ,
    INDEX_LM1: IntField,
    INDEX_LP0: IntField,
    q: FloatField,
    pe1: FloatField,
    pe2: FloatField,
    q4_1: FloatField,
    q4_2: FloatField,
    q4_3: FloatField,
    q4_4: FloatField,
    dp1: FloatField,
    lev: IntFieldIJ,
):
    """
    Args:
        km (in):
        not_exit_loop (in/temp):
        LM1 (in/temp):
        LP0 (in/temp):
        q (in/out):
        pe1 (in):
        pe2 (in):
        q4_1 (in):
        q4_2 (in):
        q4_3 (in):
        q4_4 (in):
        dp1 (in):
        lev (inout):
    """

    # This computation creates a IntField that allows for "absolute" references in the k-dimension
    # for q and pe1.

    # INDEX_LM1 and INDEX_LP0 is initialized such that if it's plugged into "q" (ex: q[0,0,INDEX_LM1]), the k level in
    # q is "k = 0".

    # For example, during the stencil computation at k = 2, INDEX_LM1[i,j,2] = -2
    with computation(FORWARD):
        with interval(0,1):
            INDEX_LM1 = 0
            INDEX_LP0 = 0
        with interval(1,None):
            INDEX_LM1 = INDEX_LM1[0,0,-1] - 1
            INDEX_LP0 = INDEX_LP0[0,0,-1] - 1

    # TODO: Can we make lev a 2D temporary?
    with computation(FORWARD), interval(...):
        LM1 = 1
        LP0 = 1
        not_exit_loop = True
        while(LP0 <= km and not_exit_loop):
            if(pe1[0,0,INDEX_LP0] < pe2):
                LP0 = LP0 + 1
                INDEX_LP0 = INDEX_LP0 + 1
            else:
                not_exit_loop = False

        LM1 = max(LP0-1,1)
        INDEX_LM1 = INDEX_LM1 + (LM1 - 1)
        LP0 = min(LP0, km)

        if(LP0 == 1):
            INDEX_LP0 = INDEX_LM1
        elif(LP0 <= km):
            INDEX_LP0 = INDEX_LM1+1
        else:
            INDEX_LP0 = INDEX_LM1
            
        if(LM1 == 1 and LP0 == 1):
            q_temp = q[0,0,INDEX_LM1] + (q[0,0,INDEX_LM1+1] - q[0,0,INDEX_LM1]) * (pe2 - pe1[0,0,INDEX_LM1]) \
                                    / (pe1[0,0,INDEX_LM1+1] - pe1[0,0,INDEX_LM1])

        elif(LM1 == km and LP0 == km):
            q_temp = q[0,0,INDEX_LM1] + (q[0,0,INDEX_LM1] - q[0,0,INDEX_LM1-1]) * (pe2 - pe1[0,0,INDEX_LM1]) \
                                    / (pe1[0,0,INDEX_LM1] - pe1[0,0,INDEX_LM1-1])

        elif(LM1 == 1 or LP0 == km):
            q_temp = q[0,0,INDEX_LP0] + (q[0,0,INDEX_LM1] - q[0,0,INDEX_LP0]) * (pe2 - pe1[0,0,INDEX_LP0]) \
                                    / (pe1[0,0,INDEX_LM1] - pe1[0,0,INDEX_LP0])
            
        else:
            while(pe2 < pe1[0,0,lev] or pe2 > pe1[0,0,lev+1]):
                lev = lev + 1
            pl = (pe2 - pe1[0, 0, lev]) / dp1[0, 0, lev]
            if pe2[0, 0, 1] <= pe1[0, 0, lev + 1]:
                pr = (pe2[0, 0, 1] - pe1[0, 0, lev]) / dp1[0, 0, lev]
                q_temp = (
                    q4_2[0, 0, lev]
                    + 0.5
                    * (q4_4[0, 0, lev] + q4_3[0, 0, lev] - q4_2[0, 0, lev])
                    * (pr + pl)
                    - q4_4[0, 0, lev] * 1.0 / 3.0 * (pr * (pr + pl) + pl * pl)
                )
            else:
                qsum = (pe1[0, 0, lev + 1] - pe2) * (
                    q4_2[0, 0, lev]
                    + 0.5
                    * (q4_4[0, 0, lev] + q4_3[0, 0, lev] - q4_2[0, 0, lev])
                    * (1.0 + pl)
                    - q4_4[0, 0, lev] * 1.0 / 3.0 * (1.0 + pl * (1.0 + pl))
                )
                lev = lev + 1
                while pe1[0, 0, lev + 1] < pe2[0, 0, 1]:
                    qsum += dp1[0, 0, lev] * q4_1[0, 0, lev]
                    lev = lev + 1
                dp = pe2[0, 0, 1] - pe1[0, 0, lev]
                esl = dp / dp1[0, 0, lev]
                qsum += dp * (
                    q4_2[0, 0, lev]
                    + 0.5
                    * esl
                    * (
                        q4_3[0, 0, lev]
                        - q4_2[0, 0, lev]
                        + q4_4[0, 0, lev] * (1.0 - (2.0 / 3.0) * esl)
                    )
                )
                q_temp = qsum / (pe2[0, 0, 1] - pe2)

        lev = lev - 1

        q = q_temp

class MapSingle:
    """
    Fortran name is map_single, test classes are Map1_PPM_2d, Map_Scalar_2d
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        kord: int,
        mode: int,
        dims: Sequence[str],
    ):
        orchestrate(
            obj=self,
            config=stencil_factory.config.dace_config,
        )

        # TODO: consider refactoring to take in origin and domain
        grid_indexing = stencil_factory.grid_indexing

        def make_quantity():
            return quantity_factory.zeros(
                [X_DIM, Y_DIM, Z_DIM],
                units="unknown",
                dtype=Float,
            )

        self._dp1 = make_quantity()
        self._q4_1 = make_quantity()
        self._q4_2 = make_quantity()
        self._q4_3 = make_quantity()
        self._q4_4 = make_quantity()
        self._tmp_qs = quantity_factory.zeros(
            [X_DIM, Y_DIM],
            units="unknown",
            dtype=Float,
        )
        self._lev = quantity_factory.zeros([X_DIM, Y_DIM], units="", dtype=int)

        self._copy_stencil = stencil_factory.from_dims_halo(
            copy_defn,
            compute_dims=dims,
        )

        self._set_dp = stencil_factory.from_dims_halo(
            set_dp,
            compute_dims=dims,
        )

        self._remap_profile = RemapProfile(
            stencil_factory,
            quantity_factory,
            kord,
            mode,
            dims=dims,
        )

        self._lagrangian_contributions = stencil_factory.from_dims_halo(
            lagrangian_contributions,
            compute_dims=dims,
        )

        self._INDEX_LM1 = quantity_factory.zeros([X_DIM, Y_DIM], units="", dtype=int)
        self._INDEX_LP0 = quantity_factory.zeros([X_DIM, Y_DIM], units="", dtype=int)
        self._km = grid_indexing.domain[2]
        self._not_exit_loop = quantity_factory.zeros([X_DIM, Y_DIM], units="", dtype=bool)
        # self._q_temp = make_quantity()

    @property
    def i_extent(self):
        return self._extents[0]

    @property
    def j_extent(self):
        return self._extents[1]

    def __call__(
        self,
        q1: FloatField,
        pe1: FloatField,
        pe2: FloatField,
        qs: Optional["FloatFieldIJ"] = None,
        qmin: Float = 0.0,
    ):
        """
        Compute x-flux using the PPM method.

        Args:
            q1 (out): Remapped field on Eulerian grid
            pe1 (in): Lagrangian pressure levels
            pe2 (in): Eulerian pressure levels
            qs (in): Bottom boundary condition
            qmin (in): Minimum allowed value of the remapped field
        """

        self._copy_stencil(q1, self._q4_1)
        self._set_dp(self._dp1, pe1, self._lev)

        if qs is None:
            self._remap_profile(
                self._tmp_qs,
                self._q4_1,
                self._q4_2,
                self._q4_3,
                self._q4_4,
                self._dp1,
                qmin,
            )
        else:
            self._remap_profile(
                qs,
                self._q4_1,
                self._q4_2,
                self._q4_3,
                self._q4_4,
                self._dp1,
                qmin,
            )
        self._lagrangian_contributions(
            self._km,
            self._not_exit_loop,
            self._INDEX_LM1,
            self._INDEX_LP0,
            # self._q_temp,
            q1,
            pe1,
            pe2,
            self._q4_1,
            self._q4_2,
            self._q4_3,
            self._q4_4,
            self._dp1,
            self._lev,
        )
        return q1
