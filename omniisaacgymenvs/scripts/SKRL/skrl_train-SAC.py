import threading 
import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.trainers.torch import ParallelTrainer
from skrl.envs.torch import wrap_env
from skrl.envs.torch import load_omniverse_isaacgym_env

from skrl.utils import set_seed



# set the seed for reproducibility

set_seed(40)



# # Define the shared model (stochastic and deterministic models) for the agent using mixins.
# class Shared(GaussianMixin, DeterministicMixin, Model):
#     def __init__(self, observation_space, action_space, device, clip_actions=False,
#                  clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
#         Model.__init__(self, observation_space, action_space, device)
#         GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
#         DeterministicMixin.__init__(self, clip_actions)

#         self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
#                                  nn.ELU(),
#                                  nn.Linear(256, 128),
#                                  nn.ELU(),
#                                  nn.Linear(128, 64),
#                                  nn.ELU())

#         self.mean_layer = nn.Linear(64, self.num_actions)
#         self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

#         self.value_layer = nn.Linear(64, 1)

#     def act(self, inputs, role):
#         if role == "policy":
#             return GaussianMixin.act(self, inputs, role)
#         elif role == "value":
#             return DeterministicMixin.act(self, inputs, role)

#     def compute(self, inputs, role):
#         if role == "policy":
#             return self.mean_layer(self.net(inputs["states"])), self.log_std_parameter, {}
#         elif role == "value":
#             return self.value_layer(self.net(inputs["states"])), {}


# # Load and wrap the Omniverse Isaac Gym environment
# # env = load_omniverse_isaacgym_env(task_name="UR10Reacher",multi_threaded=False, timeout=30)
# env = load_omniverse_isaacgym_env(task_name="UR10Reacher")
# env = wrap_env(env)

# device = env.device


# # Instantiate a RandomMemory as rollout buffer (any memory can be used for this)
# memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)





# Define the models (stochastic and deterministic models) for the agents using mixins.
# - StochasticActor: takes as input the environment's observation/state and returns an action
# - DeterministicActor: takes as input the environment's observation/state and returns an action
# - Critic: takes the state and action as input and provides a value to guide the policy
class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 256),
                                 nn.ELU(),
                                 nn.Linear(256, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 1))

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
memory = RandomMemory(memory_size=8000, num_envs=env.num_envs, device=device, replacement=True)






# Instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#spaces-and-models
models_sac = {}
models_sac["policy"] = StochasticActor(env.observation_space, env.action_space, device)
# models_sac["value"] = models_sac["policy"]  # same instance: shared model

models_sac["critic_1"] = Critic(env.observation_space, env.action_space, device)
models_sac["critic_2"] = Critic(env.observation_space, env.action_space, device)
models_sac["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
models_sac["target_critic_2"] = Critic(env.observation_space, env.action_space, device)

for model in models_sac.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#configuration-and-hyperparameters
cfg_sac = SAC_DEFAULT_CONFIG.copy()
cfg_sac["gradient_steps"] = 1  
cfg_sac["batch_size"] = 256
cfg_sac["discount_factor"] = 0.99  
#cfg_sac["polyak"] = 0.005
#cfg_sac["actor_learning_rate"] = 1e-3
#cfg_sac["critic_learning_rate"] = 1e-3
#cfg_sac["learning_rate_scheduler"] = None
#cfg_sac["learning_rate_scheduler_kwargs"] = {}
cfg_sac["random_timesteps"] = 0
cfg_sac["learning_starts"] = 200
#cfg_sac["grad_norm_clip"] = 1.0
cfg_sac["learn_entropy"] = True
#cfg_sac["entropy_learning_rate"] = 1e-3
#cfg_sac["initial_entropy_value"] = 0.2
#cfg_sac["entropy_loss_scale"] = 0.0
#cfg_sac["value_loss_scale"] = 2.0
#cfg_sac["target_entropy"] = None
cfg_sac["rewards_shaper"] = lambda rewards, timestep, timesteps: rewards * 0.01
cfg_sac["state_preprocessor"] = RunningStandardScaler
cfg_sac["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# cfg_sac["value_preprocessor"] = RunningStandardScaler
# cfg_sac["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints each 16 and 80 timesteps respectively
cfg_sac["experiment"]["write_interval"] = 16
cfg_sac["experiment"]["checkpoint_interval"] = 200
cfg_sac["experiment"]["wandb"] = True



agent = SAC(models=models_sac,
            memory=memory,
            cfg=cfg_sac,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 16000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
#trainer = ParallelTrainer(cfg=cfg_trainer, env=env, agents=agent)


#agent.load("/RLrepo/OmniIsaacGymEnvs-UR10Reacher/omniisaacgymenvs/runs/23-04-14_14-28-27-412303_PPO/checkpoints/agent_80.pt")

# start training
trainer.train()



# threading.Thread(target=trainer.train).start()

# env.run()

