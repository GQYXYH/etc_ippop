from custom_envs.mpe.scenarios import simple_spread_pri
env = simple_spread_pri.parallel_env(N=3, penalty_ratio=0,max_cycles=25,
                    full_comm=False, local_ratio=0.1, continuous_actions=True,
                    delay = 1,packet_drop_prob=0.9,
                    bandwidth_limit = 10)

env.reset(50000)
# for i in range(3):
common_actions = [0.34618735, 0.51146454,0]

# Create a dictionary where each agent's action is this common action array
actions_env = {agent: common_actions for agent in env.agents}
for index, agent in enumerate(env.agents):
    if index == 1:  # Assuming the second agent is at index 1
        # Copy the common actions and modify the last element
        modified_actions = common_actions[:]
        modified_actions[-1] = 0.8
        actions_env[agent] = modified_actions
    elif index == 2:  # Assuming the second agent is at index 1
        # Copy the common actions and modify the last element
        modified_actions = common_actions[:]
        modified_actions[-1] = 0.7
        actions_env[agent] = modified_actions
    else:
        actions_env[agent] = common_actions.copy()
# print("actions_env",actions_env)
obs, rewards, dones, infos,_ = env.step(actions_env)
# for info in infos:
#     for agent_info in info.values():
#         print(agent_info['comms'])
print("obs",obs)


for index, agent in enumerate(env.agents):
    if index == 1:  # Assuming the second agent is at index 1
        # Copy the common actions and modify the last element
        modified_actions = common_actions[:]
        modified_actions[-1] = 0.7
        actions_env[agent] = modified_actions
    elif index == 2:  # Assuming the second agent is at index 1
        # Copy the common actions and modify the last element
        modified_actions = common_actions[:]
        modified_actions[-1] = 0.9
        actions_env[agent] = modified_actions
    else:
        actions_env[agent] = common_actions.copy()
# print("actions_env",actions_env)

obs, rewards, dones, infos,_ = env.step(actions_env)
print("obs",obs)
# print("actions_env",actions_env)

obs, rewards, dones, infos,_ = env.step(actions_env)
print("obs",obs)
obs, rewards, dones, infos,_ = env.step(actions_env)
print("obs",obs)
obs, rewards, dones, infos,_ = env.step(actions_env)
print("obs",obs)

# obs, rewards, dones, infos,_ = env.step(actions_env)
# obs, rewards, dones, infos,_ = env.step(actions_env)
# obs, rewards, dones, infos,_ = env.step(actions_env)
# obs, rewards, dones, infos,_ = env.step(actions_env)

# print("obs",obs)

# obs, rewards, dones, infos,_ = env.step(actions_env)
# print("obs",obs)
# actions_env = {'agent_0': array([ 0.53047895, -1.        ,  0.02827301,  0.9128274 ,  1.        ],
#       dtype=float32), 'agent_1': array([-0.7873955 , -0.46165505, -1.        ,  0.19557247,  1.        ],
#       dtype=float32), 'agent_2': array([-0.5264923 ,  1.        , -0.97658813,  0.21012053,  1.        ],
#       dtype=float32), 'agent_3': array([-0.46580192, -0.49683416, -0.09407247,  1.        ,  0.        ],
#       dtype=float32), 'agent_4': array([0.7747581, 0.7424295, 1.       , 1.       , 0.       ],
#       dtype=float32), 'agent_5': array([ 1.       ,  0.5789777, -0.7459386,  1.       ,  1.       ],
#       dtype=float32)}
# # obs, rewards, dones, infos = env.step(actions_env)
# from pettingzoo.sisl import multiwalker_v9

# env = multiwalker_v9.env(render_mode="human")
# env.reset(seed=42)

# for agent in env.agent_iter():
#     observation, reward, termination, truncation, info = env.last()

#     if termination or truncation:
#         action = None
#     else:
#         # this is where you would insert your policy
#         action = env.action_space(agent).sample()
#     print(action)

#     env.step(action)
# env.close()