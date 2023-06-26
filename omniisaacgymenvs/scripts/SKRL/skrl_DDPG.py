import threading 
import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.resources.noises.torch import GaussianNoise, OrnsteinUhlenbeckNoise
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.trainers.torch import ParallelTrainer
from skrl.envs.torch import wrap_env
from skrl.envs.torch import load_omniverse_isaacgym_env

from skrl.utils import set_seed



# set the seed for reproducibility

set_seed(44)



# Define the models (stochastic and deterministic models) for the agents using mixins.
# - StochasticActor: takes as input the environment's observation/state and returns an action
# - DeterministicActor: takes as input the environment's observation/state and returns an action
# - Critic: takes the state and action as input and provides a value to guide the policy


class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),
                                 nn.ELU(),
                                 nn.Linear(64, self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),
                                 nn.ELU(),
                                 nn.Linear(64, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}


# # Load and wrap the Isaac Gym environment
# env = load_isaacgym_env_preview4(task_name="Cartpole")   # preview 3 and 4 use the same loader
# env = wrap_env(env)

# device = env.device


# # Instantiate a RandomMemory (without replacement) as shared experience replay memory
# memory = RandomMemory(memory_size=8000, num_envs=env.num_envs, device=device, replacement=True)





# Load and wrap the Omniverse Isaac Gym environment
# env = load_omniverse_isaacgym_env(task_name="UR10Reacher",multi_threaded=False, timeout=30)
env = load_omniverse_isaacgym_env(task_name="UR10Reacher")
env = wrap_env(env)

device = env.device


# Instantiate a RandomMemory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=100000, num_envs=env.num_envs, device=device, replacement=True)






# TD3 requires 6 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.td3.html#spaces-and-models


models_ddpg = {}
models_ddpg["policy"] = DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True)
models_ddpg["target_policy"] = DeterministicActor(env.observation_space, env.action_space, device, clip_actions=True)
models_ddpg["critic"] = Critic(env.observation_space, env.action_space, device)
models_ddpg["target_critic"] = Critic(env.observation_space, env.action_space, device)


for model in models_ddpg.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


cfg_ddpg = DDPG_DEFAULT_CONFIG.copy()
cfg_ddpg["exploration"]["noise"] = OrnsteinUhlenbeckNoise(theta=0.3, sigma=0.1, base_scale=0.5, device=device) # changed theta from 0.15 to 0.3
cfg_ddpg["gradient_steps"] = 1
cfg_ddpg["batch_size"] = 512
cfg_ddpg["random_timesteps"] = 0
cfg_ddpg["learning_starts"] = 0
cfg_ddpg["state_preprocessor"] = RunningStandardScaler
cfg_ddpg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# logging to TensorBoard and write checkpoints each 25 and 1000 timesteps respectively
cfg_ddpg["experiment"]["write_interval"] = 25
cfg_ddpg["experiment"]["checkpoint_interval"] = 4000
cfg_ddpg["experiment"]["wandb"] = True




agent = DDPG(models=models_ddpg,
            memory=memory,
            cfg=cfg_ddpg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 160000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
#trainer = ParallelTrainer(cfg=cfg_trainer, env=env, agents=agent)


agent.load("/RLrepo/OmniIsaacGymEnvs-UR10Reacher/omniisaacgymenvs/runs/23-05-19_18-21-09-832583_DDPG/checkpoints/best_agent.pt")

# start training
# trainer.train()
trainer.eval()



# threading.Thread(target=trainer.train).start()

# env.run()

