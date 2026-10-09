"""Simple CartPole policies with the same agent(state, action_space) interface."""


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


# Add a new policy here; the evaluation loop stays the same.
AGENTS = {"random": random_agent, "heuristic": heuristic_agent}
