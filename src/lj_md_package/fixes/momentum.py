"""FixMomentum -- zero center-of-mass linear momentum every N steps."""
from __future__ import annotations
from .abc import Fix


class FixMomentum(Fix):
    """``fix ID group-ID momentum N linear``.

    Every N steps subtract the COM linear momentum, ensuring total momentum
    stays at zero (eliminates drift from rounding error accumulation).
    """

    def __init__(self, fix_id: str = "momentum", group: str = "all", N: int = 100):
        super().__init__(fix_id, group, every=int(N))

    def name(self) -> str:
        return "momentum"

    def do_end_of_step(self, state) -> None:
        state.atoms.remove_com_motion()
