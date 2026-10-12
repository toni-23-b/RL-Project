import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import random
import gymnasium as gym
import matplotlib.pyplot as plt

from collections import deque
from pathlib import Path
SEED = 42

NUM_EPISODES = 3000
MAX_STEPS = 500

GAMMA = 0.99
LEARNING_RATE = 0.001
BATCH_SIZE = 64
BUFFER_SIZE = 100000

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 50000

TARGET_UPDATE_FREQ = 1000

MIN_POLE_LENGTH = 0.4
MAX_POLE_LENGTH = 1.8

TERMINAL_PENALTY = 1.0  # Additional penalty on failure; tune only as a documented experiment.
SIGMA = 0.1

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            state,
            action,
            reward,
            next_state,
            done
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(
            np.array(states), dtype=torch.float32, device=device
        )
        actions = torch.tensor(
            actions, dtype=torch.long, device=device
        ).unsqueeze(1)
        rewards = torch.tensor(
            rewards, dtype=torch.float32, device=device
        )
        next_states = torch.tensor(
            np.array(next_states), dtype=torch.float32, device=device
        )
        dones = torch.tensor(
            dones, dtype=torch.float32, device=device
        )

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

def get_epsilon(step):
    fraction = min(step / EPSILON_DECAY_STEPS, 1.0)

    return EPSILON_START + fraction * (
        EPSILON_END - EPSILON_START
    )


def select_action(state, policy_net, epsilon):
    if random.random() < epsilon:
        return random.randrange(2)

    state_tensor = torch.tensor(
        state, dtype=torch.float32, device=device
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = policy_net(state_tensor)

    return q_values.argmax(dim=1).item()

def update_dqn(policy_net, target_net, optimizer, replay_buffer):

    if len(replay_buffer) < BATCH_SIZE:
        return None

    states, actions, rewards, next_states, dones = (
        replay_buffer.sample(BATCH_SIZE)
    )

    q_values = policy_net(states)
    current_q = q_values.gather(1, actions).squeeze(1)

    with torch.no_grad():
        next_q = target_net(next_states).max(dim=1).values
        target_q = rewards + GAMMA * (1 - dones) * next_q

    loss = F.mse_loss(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()