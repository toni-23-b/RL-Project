"""Simple CartPole policies with the same agent(state, action_space) interface."""

from pathlib import Path

import torch
import torch.nn as nn


WEIGHTS_DIR = Path(__file__).parent / "weights"


class QNetwork(nn.Module):
    """The 'standard' network from the supplied test script."""

    def __init__(self, state_dim=4, action_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def random_agent(state, action_space):
    """Choose left or right uniformly, as in Assignment 2."""
    return action_space.sample()


def heuristic_agent(state, action_space):
    """Push towards the pole's predicted tilt, using a fixed 0.1 s lookahead."""
    cart_position, cart_velocity, pole_angle, pole_velocity = state
    # Angular velocity lets us react before the pole tilts too far.
    # Positive tilt means push right (1); otherwise push left (0).
    predicted_angle = pole_angle + 0.1 * pole_velocity
    return int(predicted_angle > 0)


def dqn_agent(weights_file):
    """Greedy policy from trained DQN weights, as in the supplied test script."""
    q_network = QNetwork()
    q_network.load_state_dict(torch.load(WEIGHTS_DIR / weights_file, weights_only=True))
    q_network.eval()

    def agent(state, action_space):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            return q_network(state).argmax().item()

    return agent

# Add new agents here
AGENTS = {
    "random": random_agent,
    "heuristic": heuristic_agent,
    "baseline": dqn_agent("baseline.pth"),
    "disturbance": dqn_agent("disturbance.pth"),
}
