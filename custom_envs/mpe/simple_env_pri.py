import os
import time

import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import seeding

from pettingzoo import AECEnv
from pettingzoo.mpe._mpe_utils.core import Agent
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector


alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def make_env(raw_env):
    def env(**kwargs):
        env = raw_env(**kwargs)
        env = wrappers.OrderEnforcingWrapper(env)
        return env

    return env


class SimpleEnv(AECEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
        "render_fps": 10,
    }

    def __init__(
        self,
        scenario,
        world,
        max_cycles,
        render_mode=None,
        continuous_actions=False,
        local_ratio=None,
        num_landmarks = 3
    ):
        super().__init__()

        self.render_mode = render_mode
        pygame.init()
        self.viewer = None
        self.width = 700
        self.height = 700
        self.screen = pygame.Surface([self.width, self.height])
        self.max_size = 1
        self.game_font = pygame.freetype.Font(
            os.path.join(os.path.dirname(__file__), "secrcode.ttf"), 24
        )

        # Set up the drawing window

        self.renderOn = False
        self.seed()

        self.max_cycles = max_cycles
        self.scenario = scenario
        self.world = world
        self.continuous_actions = continuous_actions
        self.local_ratio = local_ratio
        self.num_landmarks = num_landmarks
        self.current_comm_index = 0
        self.com_signal=0

        self.scenario.reset_world(self.world)

        self.agents = [agent.name for agent in self.world.agents]
        for agent in self.world.agents:
            agent.action.c = np.zeros(1)
        self.possible_agents = self.agents[:]
        self._index_map = {
            agent.name: idx for idx, agent in enumerate(self.world.agents)
        }
        self.message_queue = []
        self.prev_comm_signals = None
        self.send_message = None
        self.priority_buffer = []

        self._agent_selector = agent_selector(self.agents)

        # set spaces
        self.action_spaces = dict()
        self.observation_spaces = dict()
        state_dim = 0
        for agent in self.world.agents:
            if agent.movable and self.continuous_actions:
                space_dim = self.world.dim_p
            elif self.continuous_actions:
                space_dim = 2
            else:
                space_dim = 1
            if not agent.silent:
                space_dim += self.world.dim_c

            obs_dim = (2+2+self.num_landmarks*2+self.num_agents*4+2)
            # print(self.num_agents)
            state_dim += obs_dim

            self.action_spaces[agent.name] = spaces.Box(
                    low=np.array([-1, -1,  0]),   # 前四维是 -1，最后一维是 0
                    high=np.array([1, 1, 1]),     # 前四维是 1，最后一维是 1
                    dtype=np.float32
                )

            self.observation_spaces[agent.name] = spaces.Box(
                low=-np.float32(np.inf),
                high=+np.float32(np.inf),
                shape=(obs_dim,),
                dtype=np.float32,
            )

        self.state_space = spaces.Box(
            low=-np.float32(np.inf),
            high=+np.float32(np.inf),
            #shape=(state_dim + (space_dim - 1) * len(self.world.agents),),
            shape=(state_dim,),
            dtype=np.float32,
        )

        self.current_actions = [None] * self.num_agents
        self.count_obs = 0
        self.loss_check = np.random.rand()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def seed(self, seed=None):
      if seed is None:
          np.random.seed(1)
      else:
          np.random.seed(seed)

    def observe(self, agent):
        
        # print(self.world.agents[self._index_map[agent]],  self.steps)
        # if self.steps%2 == 1:
        self.send_message = self.process_pri_buffer(self.steps)
        if self.loss_check>=self.scenario.packet_drop_prob:
            pass
        else:
            self.send_message = None
        if self.send_message is None or all(signal == 0 or signal is None for signal in self.send_message):
            max_comm_agent = self.current_comm_index % self.num_agents
            self.send_message = np.random.rand(self.num_agents) 

            self.send_message[max_comm_agent] += 1

            self.send_message /= np.max(self.send_message)
        # else:
        #     if self.send_message is None or all(signal == 0 or signal is None for signal in self.send_message):
        #         max_comm_agent = self.current_comm_index % self.num_agents
        #         self.send_message = np.random.rand(self.num_agents) 

        #         self.send_message[max_comm_agent] += 1

        #         self.send_message /= np.max(self.send_message)
        
        
        # print(self.steps)
        # print("call call call")
        # print("self.send_message",self.send_message)
        return self.scenario.observation(
            self.world.agents[self._index_map[agent]], self.world, self.steps,self.loss_check,self.send_message
        ).astype(np.float32)

    def state(self):
        # pass
        states = tuple(
            self.scenario.observation(
                self.world.agents[self._index_map[agent]], self.world, None
            ).astype(np.float32)
            for agent in self.possible_agents
        )
        return np.concatenate(states, axis=None)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed=seed)
        self.scenario.reset_world(self.world)

        self.agents = self.possible_agents[:]
        self.rewards = {name: 0.0 for name in self.agents}
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}
        self.infos['comms'] = 0
        self.infos['frames'] = 0
        

        self.agent_selection = self._agent_selector.reset()
        self.steps = 0

        self.current_actions = [None] * self.num_agents
        self.current_comm_index = 0
        self.com_signal=0
        self.prev_comm_signals = None
        self.send_message = None
        self.priority_buffer = []
    
    def _execute_world_step(self):
        # set action for each agent
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            self._set_action(action, agent,
                             self.action_spaces[agent.name])
            # agent.action = agent.action_callback(agent,self.steps, self.world)
            agent.action = agent.action_callback(agent)
            
        
        self.world.step()
        

        global_reward = 0.0
        if self.local_ratio is not None:
            global_reward = float(self.scenario.global_reward(self.world))

        for agent in self.world.agents:
            agent_reward = float(self.scenario.reward(agent, self.world, global_reward))
            if self.local_ratio is not None:
                reward = (
                    global_reward * (1 - self.local_ratio)
                    + agent_reward * self.local_ratio
                )
            else:
                reward = agent_reward

            self.rewards[agent.name] = reward
    def update_comm_signals(self):
        self.prev_comm_signals = [other.action.c for other in self.world.agents]
        # print(self.prev_comm_signals)
    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        if agent.movable:
            # physical action
            agent.action.u = np.zeros(self.world.dim_p)
            if self.continuous_actions:
                # Process continuous action as in OpenAI MPE
                agent.action.u[0] = action[0]
                agent.action.u[1] = action[1]
                agent.action.c = action[-1]
                # print(agent.action.u[0],agent.action.u[1],agent.action.c)
            
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
        
    def send_pri(self, message, deliver_at_step):
        """Queue messages with specified delay if they are not dropped, avoiding duplicates."""
        # Check if there's already a message for this agent at the given step
        if not any(msg[1] == deliver_at_step  for msg in self.priority_buffer):
            self.priority_buffer.append((np.copy(message), deliver_at_step))
            # print("here",self.message_buffer,len(self.message_buffer))
            # print(f"Message scheduled for {agent.name} at step {deliver_at_step}")
        else:
            pass
    
    def process_pri_buffer(self, current_step):
        """Process message buffer to deliver messages that are due and target the specified agent."""
        received_message = None  # Initialize with None to handle no messages received case
        i = 0  # Initialize index counter
        # print("len(self.priority_buffer)",len(self.priority_buffer))
        while i < len(self.priority_buffer):
            message, deliver_at_step = self.priority_buffer[i]
            if deliver_at_step == current_step:
                # If message is for the current agent and step, process and remove it
                received_message = message
                
            i+=1
        if len(self.priority_buffer)>=self.max_cycles:
            self.priority_buffer.pop(0)
        # Return the received message, or a default zero array if none was received
        return received_message if received_message is not None else [0] * (self.num_agents)
    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        cur_agent = self.agent_selection
        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        self.agent_selection = self._agent_selector.next()

        self.current_actions[current_idx] = action

        if next_idx == 0:
            self.infos['comms'] = 0
            # self.reset_communication_counters()
            # for i, agent in enumerate(self.world.agents):
            #     agent.counter()
            self.scenario.reset_communication_counters()
            if self.prev_comm_signals is None or all(signal == 0 or signal is None for signal in self.prev_comm_signals):
                max_comm_agent = self.current_comm_index % self.num_agents
                self.prev_comm_signals = np.random.rand(self.num_agents) 

                self.prev_comm_signals[max_comm_agent] += 1

                self.prev_comm_signals /= np.max(self.prev_comm_signals)
            self.send_pri(message=self.prev_comm_signals,deliver_at_step=self.steps+self.scenario.delay+1)
            
            
            self.current_comm_index+=1
            self._execute_world_step()
            self.update_comm_signals()
            # print("self.prev_comm_signals",self.prev_comm_signals)
            # print("self.priority_buffer",self.priority_buffer)
            self.steps += 1
            self.infos['frames'] = self.steps
            if self.steps >= self.max_cycles:
                for a in self.agents:
                    self.terminations[a] = True
        else:
            self._clear_rewards()

        self._cumulative_rewards[cur_agent] = 0
        self._accumulate_rewards()
        


        if self.render_mode == "human":
            self.render()

    def enable_render(self, mode="human"):
        if not self.renderOn and mode == "human":
            self.screen = pygame.display.set_mode(self.screen.get_size())
            self.renderOn = True

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        self.enable_render(self.render_mode)

        self.draw()
        if self.render_mode == "rgb_array":
            observation = np.array(pygame.surfarray.pixels3d(self.screen))
            return np.transpose(observation, axes=(1, 0, 2))
        elif self.render_mode == "human":
            pygame.display.flip()
            return

    def draw(self):
        # clear screen
        pygame.event.get()
        self.screen.fill((255, 255, 255))

        # update bounds to center around agent
        all_poses = [entity.state.p_pos for entity in self.world.entities]
        cam_range = np.max(np.abs(np.array(all_poses)))

        # update geometry and text positions
        text_line = 0
        for e, entity in enumerate(self.world.entities):
            # geometry
            x, y = entity.state.p_pos
            
            y *= (
                -1
            )  # this makes the display mimic the old pyglet setup (ie. flips image)
            x = (
                (x / cam_range) * self.width // 2 * 0.9
            )  # the .9 is just to keep entities from appearing "too" out-of-bounds
            y = (y / cam_range) * self.height // 2 * 0.9
            x += self.width // 2
            y += self.height // 2
            pygame.draw.circle(
                self.screen, entity.color * 200, (x, y), entity.size * 350 / (cam_range)
            )  # 350 is an arbitrary scale factor to get pygame to render similar sizes as pyglet
            pygame.draw.circle(
                self.screen, (0, 0, 0), (x, y), entity.size * 350 / (cam_range), 1
            )  # borders
            assert (
                0 < x < self.width and 0 < y < self.height
            ), f"Coordinates {(x, y)} are out of bounds."

            # text
            if isinstance(entity, Agent):
                if entity.silent:
                    continue
                if np.all(entity.state.c == 0):
                    word = "_"
                elif self.continuous_actions:
                    word = (
                        "[" +
                        ",".join(
                            [f"{comm:.2f}" for comm in entity.state.c]) + "]"
                    )
                else:
                    word = alphabet[np.argmax(entity.state.c)]

                message = entity.name + " sends " + word + "   "
                message_x_pos = self.width * 0.05
                message_y_pos = self.height * 0.95 - \
                    (self.height * 0.05 * text_line)
                self.game_font.render_to(
                    self.screen, (message_x_pos,
                                  message_y_pos), message, (0, 0, 0)
                )
                text_line += 1

    def close(self):
        if self.renderOn:
            pygame.event.pump()
            pygame.display.quit()
            pygame.quit()
            self.renderOn = False
