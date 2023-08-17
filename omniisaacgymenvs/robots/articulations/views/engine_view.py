# To be refined


from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView


class EngineView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "EngineView",
        track_contact_forces=False,
        prepare_contact_sensors=False,
    ) -> None:

        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
            reset_xform_properties=False
        )

        # self._col = RigidPrimView(prim_paths_expr="/World/envs/.*/box", name="boxx_view", reset_xform_properties=False)
        self._startor = RigidPrimView(prim_paths_expr="/World/envs/.*/Engine/stator_instant_Scaled", name="startor_view", reset_xform_properties=False)

    # def initialize(self, physics_sim_view):
    #     super().initialize(physics_sim_view)

    #stator_instant_Scaled/Startor_Progress