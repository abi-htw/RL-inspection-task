import os 
import sys
import carb
import omni
import torch
import random
from omni.isaac.kit import SimulationApp


# This sample loads a usd stage and starts simulation
CONFIG = {"width": 1280, "height": 720, "sync_loads": True, "headless": False, "renderer": "RayTracedLighting"}

kit = SimulationApp(launch_config=CONFIG)

from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
from omni.isaac.core.prims import GeometryPrim
from omni.isaac.core.prims import RigidPrim




import numpy as np 

# from omniisaacgymenvs.robots.articulations.views.engine_view import EngineView
# from omniisaacgymenvs.robots.articulations.engine import Engine
# from omni.isaac.core.objects import VisualCuboid, DynamicCuboid
# from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
# from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units


my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

from omni.isaac.universal_robots.controllers import RMPFlowController
from omni.isaac.core import World
from followtaskUR import FollowTarget
my_task = FollowTarget(name="follow_target_task", attach_gripper=True)
my_world.add_task(my_task)
my_world.reset()

task_params = my_world.get_task("follow_target_task").get_params()
ur10_name = task_params["robot_name"]["value"]
target_name = task_params["target_name"]["value"]
my_ur10 = my_world.scene.get_object(ur10_name)
my_controller = RMPFlowController(name="target_follower_controller", robot_articulation=my_ur10, attach_gripper=True)
articulation_controller = my_ur10.get_articulation_controller()





# Locate Isaac Sim assets folder to load sample
from omni.isaac.core.utils.nucleus import get_assets_root_path, is_file

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    kit.close()
    sys.exit()
usd_path = "/inst_assets/bmw_engine/ENGINE_V3/Collected_Startor_29.8/engine_inst.usd"
box_usd = "/inst_assets/table_usd/table_4.usd"
glass_wall = "/inst_assets/Glass_wall/glass_wall2.usd"
add_reference_to_stage(usd_path=box_usd, prim_path="/World/box")
add_reference_to_stage(usd_path=glass_wall, prim_path="/World/glass_wall")
add_reference_to_stage(usd_path=usd_path, prim_path="/World/Engine")




# omni.usd.get_context().open_stage(usd_path)
articulated_system_2 = my_world.scene.add(Robot(prim_path="/World/box", name="my_box", scale= torch.tensor([0.01, 0.01, 0.01]), position=  torch.tensor([0.0, 0.0, 0.0])))
articulated_system_3 = my_world.scene.add(Robot(prim_path="/World/glass_wall", name="my_glass", position=  torch.tensor([0.0, 0.0, 0.9])))

# articulated_system_1 = my_world.scene.add(Robot(prim_path="/World/Engine", name="my_engine", scale= torch.tensor([0.001, 0.001, 0.001]), position=  torch.tensor([0.4, 0.0, 0.67])))
articulated_system_1 = my_world.scene.add(RigidPrim(prim_path="/World/Engine", name="my_engine", scale= torch.tensor([0.001, 0.001, 0.001]), position=  torch.tensor([0.4, 0.0, 0.67])))

from omni.isaac.core.objects import VisualCuboid, DynamicCuboid
from omni.isaac.core.objects import VisualSphere




# cube_1 = my_world.scene.add(
#     VisualCuboid(
#         prim_path="/new_cube_1",
#         name="visual_cube",
#         position=np.array([0.5, 0,0.77]),
#         size=0.1,
#         color=np.array([255, 255, 255]),
#     )
# )

# sphere_1 = my_world.scene.add(VisualSphere(prim_path="/new_sphere_1",
#                                            name= "visual_sphere",
#                                            radius=0.3,
#                                            position=[0.4,0.0,0.67],
#                                              color=np.array([300, 400, 800]), 
                                           
#                                            ))

def end_effector_check(end_pos= [0.5, 0,0.77],engine_pos=[0.4, 0.0, 0.67]):
        """ Function to check if the end effector is in the region of the engine"""
        
        radius_inner = 0.3
        diff = torch.tensor(end_pos, device="cuda:0") -  torch.tensor((engine_pos),device = "cuda:0")
        # print(diff)
        
        dist = torch.norm(diff,dim=1)
        # dist2 = torch.norm(self.relative_pos -  torch.tensor((engine_pos)) )
        # print(dist2)
        # print(self.relative_pos)
        # print(f"eepos{end_pos}")
        print(f"End effector distance : {torch.abs(dist)}")
        if_in_region = torch.where(torch.abs(dist) < radius_inner, True, False)  
        print(if_in_region)
        return if_in_region
    
    
def safety_check(end_pos):
    print(end_pos[0][1])
    print(end_pos[0][0])
    
    left_right = lambda g : False if g[0][1] < 0.50 and g[0][1] > -0.50 else True
    front_back = lambda g : False if g[0][0] < 0.85 and g[0][0] > -0.85 else True
    return left_right(end_pos), front_back(end_pos)
    
def get_rand_eng_layer_pos_sphere_layer(pos= [0.4, 0.0, 0.67], n_reset_envs=4):
    
        radius_outer = 0.2
        radius_inner = 0.1
        pos =torch.tensor(pos,)
        while True:
            x = random.uniform(pos[0]-0.3, pos[0]+0.3)
            y = random.uniform(pos[1]- 0.3, pos[1] + 0.3)
            z = random.uniform(pos[2] + 0.1, pos[2]+ 0.2)
            # print(torch.norm(torch.tensor([x,y,z])- pos))
            if torch.norm(torch.tensor([x,y,z])- pos) > radius_inner and torch.norm(torch.tensor([x,y,z])- pos) < radius_outer :
                pass
            else:
                break
        relative_pos = torch.tensor([x,y,z])
        my_world.scene.clear(VisualCuboid(
        prim_path="/new_cube_1",
        name="visual_cube",
        position=np.array([x,y,z]),
        size=0.04,
        color=np.array([255, 255, 255]),
    ))
        cube_1 = my_world.scene.add(
    VisualCuboid(
        prim_path="/new_cube_1",
        name="visual_cube",
        position=np.array([1.9, 0.4, 1.5]),
        size=0.04,
        color=np.array([255, 255, 255]),
    )
)
        end_pos = my_ur10.end_effector.get_world_pose()
        print((f"End Pos : {end_pos[0]}"))
        end_pos[0][2]-= 0.25
        end_effector_check(end_pos[0].tolist())
        
        # end_effector_check([x,y,z])
        print(f"random positions{x,y,z}")
        return [x,y,z]
        # return torch.stack([relative_pos]* n_reset_envs)
    
for j in range(9000):
    my_world.step(render=True)
    if j%100==0:
        print(j)
        pos = get_rand_eng_layer_pos_sphere_layer()
        boundary = safety_check(my_ur10.end_effector.get_world_pose())
        print(f"Boundary : front_back = {boundary[1]}, Boundary : left_right = {boundary[0]}")
        observations = my_world.get_observations()
        
        actions = my_controller.forward(
            # target_end_effector_position=observations[target_name]["position"],
            target_end_effector_orientation=observations[target_name]["orientation"],
            # target_end_effector_position= np.array(pos),
            target_end_effector_position= np.array([1.9, 0.2 , 1.5]),
            
            
        )
        articulation_controller.apply_action(actions)
        


        


kit.update()
kit.update()

print("Loading stage...")
from omni.isaac.core.utils.stage import is_stage_loading

while is_stage_loading():
    kit.update()
print("Loading Complete")
# omni.timeline.get_timeline_interface().play()

while kit.is_running():
        # Run in realtime mode, we don't specify the step size
        kit.update()
# omni.timeline.get_timeline_interface().stop()
kit.close()


# for i in range(5):
#     my_world.reset()
#     articulated_system_1.set_world_pose(position=np.array([0.0, 2.0, 0.0]) / get_stage_units())
#     for i in range(500):
#         my_world.step(render=True)
#         print(articulated_system_1.get_angular_velocity())
#         # print(cube_2.get_world_pose())

# simulation_app.close()