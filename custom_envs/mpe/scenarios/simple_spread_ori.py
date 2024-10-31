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

    def make_world(self, N=3, penalty_ratio=0.5, full_comm=False,delay = 2, packet_drop_prob=0.2,bandwidth_limit=10,landmarks=3):
        world = World()
        # set any world properties first
        world.dim_c = 2

        self.collision_penal = collision_penal
        self.vision = vision
        self.num_agents = N
        num_agents = N
        self.num_landmarks = landmarks
        num_landmarks = landmarks
        self.n_agents = N
        # self.n_collisions = 0
        world.collaborative = True
        self.full_comm = full_comm
        self.penalty_ratio = penalty_ratio 
        self.last_message = {}
        self.packet_drop_prob = packet_drop_prob
        self.delay = delay
        self.bandwidth_limit = bandwidth_limit



        self.total_sep = 1.25
        self.arena_size = 1
        self.ideal_sep = self.total_sep / (self.n_agents - 1)
        self.rewards = np.zeros(self.n_agents)
        self.state_buff = []
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
            # agent.counter = self.reset_communication_counters()
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
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
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        world.dists = []

        world.steps = 0
        self.message_buffer=[]
        

    def benchmark_data(self, agent, world):
        if agent.name == 'agent 0':
            entity_info = []
            state_shape = 0
            for l in world.landmarks:
                entity_info.append(l.state.p_pos)
                state_shape += len(l.state.p_pos)
            agent_info = []
            for a in world.landmarks:
                agent_info.append(a.state.p_pos)
                agent_info.append(a.state.p_vel)
                state_shape += (len(a.state.p_pos) + len(a.state.p_pos))
            state = np.concatenate(entity_info+agent_info)
            ret = {}
            ret['state_shape'] =state_shape
            ret['s'] = state
        return  ret

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
            world.dists = np.array([[np.linalg.norm(a.state.p_pos - l.state.p_pos) for l in world.landmarks]
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
           
            else:
                other_pos.append(other.state.p_pos - agent.state.p_pos)
                other_pos.append(other.state.p_vel - agent.state.p_vel)
                
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