# To be refined


from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView


class EngineView(RigidPrimView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "EngineView",
    ) -> None:

        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
            reset_xform_properties=False
        )

        # self._col = RigidPrimView(prim_paths_expr="/World/envs/.*/box", name="boxx_view", reset_xform_properties=False)
    # def initialize(self, physics_sim_view):
    #     super().initialize(physics_sim_view)