import numpy as np
from gymnasium.utils import EzPickle

from custom_envs.mpe.core import Agent, Landmark, World
from custom_envs.mpe.scenario import BaseScenario
from custom_envs.mpe.simple_env_rp import SimpleEnv, make_env

from pettingzoo.utils.conversions import parallel_wrapper_fn
def flatten_mixed_list(mixed_list):
    flat_list = []
    for item in mixed_list:
        if isinstance(item, np.ndarray):
            # Extend flat_list with array elements if the item is an array
            flat_list.extend(item.tolist())
        else:
            # Append the item itself if it's not an array
            flat_list.append(item)
    return flat_list

class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        N=3,
        penalty_ratio = 0.5,
        full_comm=True,
        local_ratio=0.5,
        max_cycles=25,
        delay = 10,
        packet_drop_prob=0.1,
        bandwidth_limit = 10,
        landmarks=3,
        continuous_actions=False,
        render_mode=None,
        ):
        EzPickle.__init__(
            self, N=N, 
            penalty_ratio=penalty_ratio,  
            local_ratio=local_ratio, 
            full_comm=full_comm,
            max_cycles=max_cycles, 
            delay = delay,packet_drop_prob=packet_drop_prob,
            bandwidth_limit = bandwidth_limit,
            continuous_actions=continuous_actions, 
            landmarks = landmarks,
            render_mode=render_mode
        )
        scenario = Scenario()
        world = scenario.make_world(N, penalty_ratio, full_comm, delay, packet_drop_prob,bandwidth_limit,landmarks)
        super().__init__(
            # self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            local_ratio=local_ratio,
            num_landmarks = landmarks
        )
        self.metadata["name"] = "simple_formulation_v2"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)
from scipy.optimize import linear_sum_assignment
collision_penal = 0
vision = 10
def get_thetas(poses):
    # compute angle (0,2pi) from horizontal
    thetas = [None]*len(poses)
    for i in range(len(poses)):
        # (y,x)
        thetas[i] = find_angle(poses[i])
    return thetas


def find_angle(pose):
    # compute angle from horizontal
    angle = np.arctan2(pose[1], pose[0])
    if angle<0:
        angle += 2*np.pi
    return angle


class Scenario(BaseScenario):
    def __init__(self):
        super().__init__()
        # self.current_time = 0
        self.message_queue = []
        self.max_queue_length = 10
        self.communication_count = 0
        self.message_buffer = []
        self.one_round=0
    def action_callback(self, agent):
       
      
        
      
      return agent.action

    def make_world(self, N=3, penalty_ratio=0.5, full_comm=False,delay = 2, packet_drop_prob=0.2,bandwidth_limit=10,landmarks=1):
        world = World()
        # set any world properties first
        world.dim_c = 2
        self.collision_penal = collision_penal
        self.vision = vision
        self.num_agents = N
        num_agents = N
        self.num_landmarks = 1
        num_landmarks = 1
        self.n_agents = N
        world.collaborative = True
        self.full_comm = full_comm
        self.penalty_ratio = penalty_ratio 
        self.last_message = {}
        self.packet_drop_prob = packet_drop_prob
        self.delay = delay
        self.bandwidth_limit = bandwidth_limit



        self.arena_size = 1

        self.rewards = np.zeros(self.n_agents)

        self.target_radius = 0.5  # fixing the target radius for now
        self.ideal_theta_separation = (2 * np.pi) / self.n_agents  # ideal theta difference between two agents
        self.dist_thres = 0.05
        self.theta_thres = 0.1


        # add agents
        self.communication_counters = {agent.name: 0 for agent in world.agents}
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = True
            agent.silent = False
            agent.size = 0.03
            agent.action_callback = self.action_callback
            self.last_message[agent.name] = np.zeros(world.dim_p + 1)
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.02
        # make initial conditions
        self.reset_world(world)

        world.dists = []
        return world

    def reset_world(self, world):
        # random properties for agents
        self.message_buffer=[]
        self.one_round=0
        self.communication_count = 0
        self.communication_counters = {agent.name: 0 for agent in world.agents}
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
            self.last_message[agent.name] = np.zeros(world.dim_p + 1)
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        world.landmarks[0].state.p_pos = np.random.uniform(-0.25, +0.25, world.dim_p)
        world.landmarks[0].state.p_vel = np.zeros(world.dim_p)


        world.steps = 0
        world.dists = []
        self.message_buffer=[]




    def is_obs(self,entity1,entity2):
        delt_pos = entity1.state.p_pos - entity2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delt_pos)))
        return True if dist < self.vision else False
    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world, global_reward=None):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        if agent.name == 'agent_0':
            rew = 0

            landmark_pose = world.landmarks[0].state.p_pos
            relative_poses = [agent.state.p_pos - landmark_pose for agent in world.agents]
            thetas = get_thetas(relative_poses)

            theta_min = min(thetas)
            self.expected_positions = [landmark_pose + self.target_radius * np.array(
                [np.cos(theta_min + i * self.ideal_theta_separation),
                 np.sin(theta_min + i * self.ideal_theta_separation)])
                              for i in range(self.n_agents)]


            world.dists = np.array([[np.linalg.norm(a.state.p_pos - pos) for pos in self.expected_positions]
                                    for a in world.agents])
            # optimal 1:1 agent-landmark pairing (bipartite matching algorithm)
            self.min_dists = self._bipartite_min_dists(world.dists)
            # the reward is normalized by the number of agents
            rew = -np.mean(self.min_dists)

            collision_rew = 0
            for b in world.agents:
                for a in world.agents:
                    if self.is_collision(a, b):
                        collision_rew -= self.collision_penal
            collision_rew /= (2 * self.n_agents)
            rew += collision_rew
            

            rew = np.clip(rew, -15, 15)
            self.rewards = np.full(self.n_agents, rew)
            world.min_dists = self.min_dists
        # print('reward = {} , return = {}'.format(self.rewards,self.rewards.mean()))
        return self.rewards.mean()

    def _bipartite_min_dists(self, dists):
        ri, ci = linear_sum_assignment(dists)
        min_dists = dists[ri, ci]
        return min_dists
    
    def global_reward(self, world):
        rew = 0.0
        # for lm in world.landmarks:
        #     dists = [
        #         np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
        #         for a in world.agents
        #     ]
        #     rew -= min(dists)
        return rew
    


    # def observation(self, agent, world,current_step,current_comm_index):
    #     # get positions of all entities in this agent's reference frame
    #     entity_pos = []
    #     for entity in world.landmarks:  # world.entities:
    #         if self.is_obs(agent,entity):
    #             entity_pos.append(entity.state.p_pos - agent.state.p_pos)
    #         else:
    #             entity_pos.append(np.zeros_like(entity.state.p_pos - agent.state.p_pos))
    #     # entity colors
    #     entity_color = []
    #     for entity in world.landmarks:  # world.entities:
    #         entity_color.append(entity.color)
    #     # communication of all other agents
        
    #     other_pos = []
    #     current_comm_index=current_comm_index-1
    #     # for i in range(self.n_agents):
    #     if agent.name != f'agent_{current_comm_index}':
    #             # print("self.current_comm_index",self.current_comm_index)
    #         # It's this walker's turn to send its data
    #             for other in world.agents:
    #                 # if j == i:
    #                 #     neighbor_obs.append(0.0)  # No data from itself
    #                 #     neighbor_obs.append(0.0)
    #                 # elif 
    #                 if other.name == f'agent_{current_comm_index}':
    #                     # Calculate relative positions for communication
    #                     other_pos.append(other.state.p_pos - agent.state.p_pos)
    #                     other_pos.append(other.state.p_vel - agent.state.p_vel)
    #                 else: 
    #                     other_pos.extend([0.0, 0.0, 0.0, 0.0])
    #     else:
    #         # print("self.current_comm_index")
    #         # Not this walker's turn, it sends zero data
    #         for j in range(self.n_agents):
    #             other_pos.extend([0.0, 0.0, 0.0, 0.0])
        
        
    #     other_pos_timer = flatten_mixed_list(other_pos)
    #     # print(len(other_pos_timer))
    #     other_pos = np.concatenate((flatten_mixed_list(other_pos),np.array([self.delay])))
    #     other_pos = np.concatenate((other_pos,[0]))
    #     # print(agent.action.c[0],agent.action.c[1])
    #     if all(value == 0 for value in other_pos_timer):
    #         other_pos[-1]+=1
    #     if np.random.rand() >= self.packet_drop_prob:#packet drops
    #         pass
    #     else:
    #         other_pos = np.zeros(4*self.n_agents+2)
    #     # print("other_pos.shape[0]",other_pos.shape[0])
    #     assert other_pos.shape[0]==(self.n_agents*4+2)
    #     agent_velocity = np.array(agent.state.p_vel)  # ensure array
    #     agent_position = np.array(agent.state.p_pos)  # ensure array
    #     entity_positions = np.array(entity_pos).flatten()  # flatten list of position arrays
    #     other_positions = np.array(other_pos)  # already an array, make sure it's flat if needed
    #     # print("other_positions1",other_positions)
    #     real_send_frame = self.delay+current_step
    #     # print()
    #     if current_step > 0:
    #         self.send_message(other_positions, real_send_frame,agent)
    #         other_positions = np.array(self.process_message_buffer(current_step,agent))
        
    #     obs = np.concatenate([agent_velocity, agent_position, entity_positions, other_positions])
    #     # print("obs",obs.shape)
    #     assert obs.shape[0]==(2+2+self.num_landmarks*2+self.n_agents*4+2)
    #     return np.concatenate([agent_velocity, agent_position, entity_positions, other_positions])
    def observation(self, agent, world,current_step,current_comm_index):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        
        
        for entity in world.landmarks:  # world.entities:
            
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        other_pos = []
        current_comm_index=current_comm_index%self.n_agents
        # print(current_comm_index)
        
        # if agent.name != f'agent_{current_comm_index}':
        #         # print("self.current_comm_index",self.current_comm_index)
        #     # It's this walker's turn to send its data
        for other in world.agents:
            if other is agent:
                other_pos.extend([0.0, 0.0, 0.0, 0.0])
            # if j == i:
            #     neighbor_obs.append(0.0)  # No data from itself
            #     neighbor_obs.append(0.0)
            # elif 
            elif current_comm_index == 2:
                if (other.name == f'agent_{0}' or other.name == f'agent_{1}') and other is not agent:
                    # Calculate relative positions for communication
                    # print("here")
                    other_pos.append(other.state.p_pos - agent.state.p_pos)
                    other_pos.append(other.state.p_vel - agent.state.p_vel)
                else: 
                    other_pos.extend([0.0, 0.0, 0.0, 0.0])
            elif current_comm_index == 4:
                if (other.name == f'agent_{2}' or other.name == f'agent_{3}') and other is not agent:
                    # Calculate relative positions for communication
                    other_pos.append(other.state.p_pos - agent.state.p_pos)
                    other_pos.append(other.state.p_vel - agent.state.p_vel)
                else: 
                    other_pos.extend([0.0, 0.0, 0.0, 0.0])
            elif current_comm_index == 6:
                if (other.name == f'agent_{4}' or other.name == f'agent_{5}') and other is not agent:
                    # Calculate relative positions for communication
                    other_pos.append(other.state.p_pos - agent.state.p_pos)
                    other_pos.append(other.state.p_vel - agent.state.p_vel)
                else: 
                    other_pos.extend([0.0, 0.0, 0.0, 0.0])
            elif current_comm_index == 0:
                if (other.name == f'agent_{6}' or other.name == f'agent_{7}') and other is not agent:
                    # Calculate relative positions for communication
                    other_pos.append(other.state.p_pos - agent.state.p_pos)
                    other_pos.append(other.state.p_vel - agent.state.p_vel)
                else: 
                    other_pos.extend([0.0, 0.0, 0.0, 0.0])
            else:
                other_pos.extend([0.0, 0.0, 0.0, 0.0])
        # else:
        #     # print("self.current_comm_index")
        #     # Not this walker's turn, it sends zero data
        #     for j in range(self.n_agents):
        #         other_pos.extend([0.0, 0.0, 0.0, 0.0])
        
        other_pos_timer = flatten_mixed_list(other_pos)
        # print("other_pos.shape[0]",other_pos_timer)
        other_pos = np.concatenate((flatten_mixed_list(other_pos),np.array([self.delay])))
        other_pos = np.concatenate((other_pos,[0]))
        # print(agent.action.c[0],agent.action.c[1])
        if all(value == 0 for value in other_pos_timer):
            other_pos[-1]+=1
        if np.random.rand() >= self.packet_drop_prob:#packet drops
            pass
        else:
            other_pos = np.zeros(4*self.n_agents+2)
        # print("other_pos.shape[0]",other_pos.shape[0])
        assert other_pos.shape[0]==(self.n_agents*4+2)
        agent_velocity = np.array(agent.state.p_vel)  # ensure array
        agent_position = np.array(agent.state.p_pos)  # ensure array
        entity_positions = np.array(entity_pos).flatten()  # flatten list of position arrays
        other_positions = np.array(other_pos)  # already an array, make sure it's flat if needed
        # print("other_positions1",other_positions)
        real_send_frame = self.delay+current_step
        # print()
        if current_step > 0:
            self.send_message(other_positions, real_send_frame,agent)
            other_positions = np.array(self.process_message_buffer(current_step,agent))
        
        obs = np.concatenate([agent_velocity, agent_position, entity_positions, other_positions])
        # print("obs",obs.shape)
        assert obs.shape[0]==(2+2+self.num_landmarks*2+self.n_agents*4+2)
        return np.concatenate([agent_velocity, agent_position, entity_positions, other_positions])
    # def send_message(self, message, real_send_frame,agent):
    #     """Queue messages with specified delay if they are not dropped."""
    #     # print(message)
    #     # print(f"Appending message with current_step {current_step}, deliver_at_step {deliver_at_step}")
    #     self.message_buffer.append((np.copy(message), real_send_frame,agent))
    #     print("here",self.message_buffer,len(self.message_buffer))
    #     # print(f"Message scheduled for {agent.name} at step {real_send_frame}")
    #     # if len(self.message_buffer) > (self.delay*self.n_agents+9):
    #     #     self.message_buffer.pop(0)
    def send_message(self, message, deliver_at_step, agent):
        """Queue messages with specified delay if they are not dropped, avoiding duplicates."""
        # Check if there's already a message for this agent at the given step
        if not any(msg[1] == deliver_at_step and msg[2] == agent for msg in self.message_buffer):
            self.message_buffer.append((np.copy(message), deliver_at_step, agent))
            # print("here",self.message_buffer,len(self.message_buffer))
            # print(f"Message scheduled for {agent.name} at step {deliver_at_step}")
        else:
            pass
            # print(f"Duplicate message prevented for {agent.name} at step {deliver_at_step}")
    # def process_message_buffer(self, current_step,agent):
    #     """Process message buffer to deliver messages that are due and target the specified agent."""
    #     received_message = None  # Changed to None to handle no messages received case
        
    #     for message, deliver_at_step, target_agent_id in self.message_buffer:
    #         # print(self.message_buffer)
    #         if deliver_at_step == current_step and target_agent_id == agent:
    #             # print(message)
                
    #             received_message = (message, deliver_at_step)
    #     if received_message:
    #         real_receive = received_message[0]
    #     else:
    #         # print("here")
    #         real_receive = np.zeros(2*self.n_agents+1)
    #     # print(real_receive)
    #     return real_receive
    def process_message_buffer(self, current_step, agent):
        """Process message buffer to deliver messages that are due and target the specified agent."""
        received_message = None  # Initialize with None to handle no messages received case
        i = 0  # Initialize index counter

        while i < len(self.message_buffer):
            message, deliver_at_step, target_agent_id = self.message_buffer[i]
            if deliver_at_step == current_step and target_agent_id == agent:
                # If message is for the current agent and step, process and remove it
                received_message = message
                self.message_buffer.pop(i)  # Remove the message from the buffer
                # Don't increment i, as the next item will shift into the current index
            else:
                i += 1  # Only increment if no item was removed

        # Return the received message, or a default zero array if none was received
        return received_message if received_message is not None else np.zeros(4*self.n_agents+2)
    
    def reset_communication_counters(self):
        for key in self.communication_counters:
            self.communication_counters[key] = 0