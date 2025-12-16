import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import numpy as np
from collections import deque

import gymnasium as gym

class DQNAgentConfig:
    env_name = "LunarLander-v3"
    
    batch_size = 64
    learning_rate = 1e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gamma = 0.99

    # 90% random actions at start
    eps_start = 0.9
    
    # 5% random actions at end
    eps_end = 0.05
    
    # how fast to stop exploring
    eps_decay = 1000

    # update rate for target net
    tau = 0.005

    replay_buffer_capacity = 100000

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) model. It is a MLP that maps the state 
    to the Q-values.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    """
    The trajectory of the Lunar Lander
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the replay buffer.
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    The agent that interacts with the environment.
    """
    def __init__(self, config: DQNAgentConfig):
        self.env = gym.make(config.env_name)
    
        self.n_observations = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.device = config.device
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate
        self.gamma = config.gamma
        self.eps_start = config.eps_start
        self.eps_end = config.eps_end
        self.eps_decay = config.eps_decay
        self.tau = config.tau

        self.policy_net = DQN(self.n_observations, 128, self.n_actions).to(self.device)
        self.target_net = DQN(self.n_observations, 128, self.n_actions).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.memory = ReplayBuffer(config.replay_buffer_capacity)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.steps_done = 0

    def select_action(self, state):
        self.steps_done += 1

        # Epsilon-greedy strategy

        # Generate a random number
        sample = random.random()

        # Compute the epsilon threshold
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)


        # Select action
        if sample > eps_threshold:
            # Exploit (ask net: what is the best action?)
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                return self.policy_net(state_tensor).max(1)[1].view(1, 1)
        else:
            # Explore (random action, so the net can explore more actions)
            return torch.tensor([[random.randrange(self.n_actions)]], dtype=torch.long).to(self.device)


    
    def optimize_model(self):
        # The model learns through trial and error

        # If we cant sample a batch, do nothing
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch of experiences
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device).unsqueeze(1)       
        reward = torch.tensor(reward, dtype=torch.float, device=self.device).unsqueeze(1)
        next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)
        done = torch.tensor(done, dtype=torch.float, device=self.device).unsqueeze(1)
        
        # The bellman equation:
        # Q-value of the current state-action pair
        # .gather(1, action) selects the Q-value of the action that was taken
        current_q_values = self.policy_net(state).gather(1, action)
        
        # Q-value of the next state-action pair
        # max(1)[0] selects the maximum Q-value
        with torch.no_grad():
            next_state_values = self.target_net(next_state).max(1)[0].unsqueeze(1)

        # Bellman Target
        # If the game ended (done=1), there is no future, so (1 - done) becomes 0.
        expected_q_values = reward + (self.gamma * next_state_values * (1 - done))

        # Minimize the difference
        criterion = nn.SmoothL1Loss() # More stable MSE
        loss = criterion(current_q_values, expected_q_values)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (prevents "exploding gradients")
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        
        self.optimizer.step()

        # Update target network
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)