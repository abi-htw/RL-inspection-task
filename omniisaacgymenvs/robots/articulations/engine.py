# To be refined

from typing import Optional

from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

import numpy as np
import torch

import carb


class Engine(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "Engine",
        usd_path: Optional[str] = None,
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
        scale: Optional[torch.tensor] = None,
    ) -> None:
        """[summary]
        """
        self._usd_path = usd_path
        self._name = name

        if self._usd_path is None:
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                carb.log_error("Could not find Isaac Sim assets folder")
            # self._usd_path = "/inst_assets/bmw_engine.usd"
            self._usd_path = "/inst_assets/bmw_engine/Startor Version 3/28.usd"

            # self._usd_path = "/inst_assets/Isaac/2022.1/Isaac/Props/Box/small_KLT.usd"



        # self._position = torch.tensor([1.1, 0.8, 0.5]) if translation is None else translation
        self._position = torch.tensor([0.6, 0.8, 0.5]) if translation is None else translation
        self._orientation = torch.tensor([0.707, -0.707, 0.0, 0.0]) if orientation is None else orientation
        self._scale = torch.tensor([0.001, 0.001, 0.001]) if scale is None else scale  # Default scale is [1, 1, 1]
        # self._scale = torch.tensor([0.01]) if scale is None else scale
        # self._orientation = torch.tensor([0.1, 0.0, 0.0, 0.0]) if orientation is None else orientation
        
    
        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            scale= self._scale,
            articulation_controller=None,
        )
