

import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.trpo import TRPO, TRPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env
from skrl.envs.torch import load_omniverse_isaacgym_env
from skrl.utils import set_seed
import psutil
import threading


# set the seed for reproducibility
set_seed(42)
# set_seed()



# Define the models (stochastic and deterministic models) for the agent using mixins.
# - Policy: takes as input the environment's observation/state and returns an action
# - Value: takes the state as input and provides a value to guide the policy
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),
                                 nn.ELU(),
                                 nn.Linear(64, self.num_actions))

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),
                                 nn.ELU(),
                                 nn.Linear(64, 1))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


# Load and wrap the Isaac Gym environment
env = load_omniverse_isaacgym_env(task_name="UR10Reacher")
# env = load_omniverse_isaacgym_env(task_name="UR10Reacher",multi_threaded= True, timeout=30)

env = wrap_env(env)

device = env.device


# Instantiate a RandomMemory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)


# Instantiate the agent's models (function approximators).
# TRPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.trpo.html#spaces-and-models
models_trpo = {}
models_trpo["policy"] = Policy(env.observation_space, env.action_space, device)
models_trpo["value"] = Value(env.observation_space, env.action_space, device)


# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.trpo.html#configuration-and-hyperparameters
cfg_trpo = TRPO_DEFAULT_CONFIG.copy()
cfg_trpo["rollouts"] = 16  # memory_size
cfg_trpo["learning_epochs"] = 8
cfg_trpo["mini_batches"] = 2
cfg_trpo["discount_factor"] = 0.99
cfg_trpo["lambda"] = 0.95
cfg_trpo["learning_rate"] = 4e-4
cfg_trpo["grad_norm_clip"] = 1.0
cfg_trpo["value_loss_scale"] = 2.0
cfg_trpo["state_preprocessor"] = RunningStandardScaler
cfg_trpo["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg_trpo["value_preprocessor"] = RunningStandardScaler
cfg_trpo["value_preprocessor_kwargs"] = {"size": 1, "device": device}
cfg_trpo["conjugate_gradient_steps"]= 20
# logging to TensorBoard and write checkpoints each 16 and 80 timesteps respectively
cfg_trpo["experiment"]["write_interval"] = 16
cfg_trpo["experiment"]["checkpoint_interval"] = 13000
cfg_trpo["experiment"]["wandb"] = True


agent = TRPO(models=models_trpo,
            memory=memory,
            cfg=cfg_trpo,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

agent.track_data("Resource / CPU usage", psutil.cpu_percent())
# Track more data such as accuracy
# print(agent.record_transition())
# agent.track_data("accuracy", env.extras)

# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 1000000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
agent.load("/RLrepo/ur10reacher/omniisaacgymenvs/runs/23-08-14_11-12-45-946128_TRPO/checkpoints/agent_988000.pt")
# agent.load("/RLrepo/ur10reacher/omniisaacgymenvs/runs/23-08-11_16-58-10-525562_TRPO/checkpoints/best_agent.pt")

# start training
trainer.train()
# trainer.eval()

# threading.Thread(target=trainer.eval).start()
# threading.Thread(target=trainer.train).start()


# env.run()

