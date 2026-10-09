"""Evaluate any policy with the Assignment 3 pole lengths and force schedule."""

import argparse
import csv
from pathlib import Path

import gym
import matplotlib.pyplot as plt
import numpy as np

from agents import AGENTS


POLE_LENGTHS = np.linspace(0.4, 1.8, 30)


def test_pole_length(env, agent, seed):
    """Adapt the provided test loop to a policy that returns an action."""
    state, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    terminated = False
    total_reward = 0

    while not terminated:
        action = agent(state, env.action_space)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if 500 <= total_reward <= 1000 and total_reward % 25 == 0:
            env.unwrapped.force_mag = 75

        if total_reward > 1000:
            env.unwrapped.force_mag = 25 + 0.01 * total_reward

    return total_reward


def evaluate_agent(agent, episodes=10, seed=42):
    """Return episode scores, with one row per pole length."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    scores = np.zeros((len(POLE_LENGTHS), episodes))

    for length_index, length in enumerate(POLE_LENGTHS):
        for episode in range(episodes):
            env = gym.make("CartPole-v1")
            env.unwrapped.length = length
            episode_seed = seed + length_index * episodes + episode
            try:
                scores[length_index, episode] = test_pole_length(
                    env, agent, episode_seed
                )
            finally:
                env.close()

        print(
            f"Pole length {length:.2f}: "
            f"{scores[length_index].mean():.1f} +/- "
            f"{scores[length_index].std():.1f} steps"
        )

    return scores


def save_results(agent_name, scores, seed, output_dir):
    """Save individual runs, per-length statistics, and the required bar plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = scores.shape[1]
    means = scores.mean(axis=1)
    stds = scores.std(axis=1)

    with (output_dir / f"{agent_name}_episodes.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent", "pole_length", "episode", "seed", "score"])
        for length_index, length in enumerate(POLE_LENGTHS):
            for episode, score in enumerate(scores[length_index]):
                writer.writerow([
                    agent_name, length, episode + 1,
                    seed + length_index * episodes + episode, score,
                ])

    with (output_dir / f"{agent_name}_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pole_length", "mean_score", "std_score", "episodes"])
        writer.writerows(zip(POLE_LENGTHS, means, stds, [episodes] * len(means)))

    fig, ax = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(POLE_LENGTHS))
    ax.bar(positions, means, yerr=stds, capsize=3, alpha=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{length:.2f}" for length in POLE_LENGTHS], rotation=45)
    ax.set_xlabel("Pole length")
    ax.set_ylabel("Episode length (steps)")
    ax.set_title(
        f"{agent_name.capitalize()} agent: "
        f"average over all pole lengths = {means.mean():.1f}"
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"{agent_name}_bar_plot.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=["all", *AGENTS], default="all")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0:
        parser.error("episodes must be at least 1 and seed must be non-negative")

    names = AGENTS if args.agent == "all" else [args.agent]
    for name in names:
        print(f"\nEvaluating {name} agent")
        scores = evaluate_agent(AGENTS[name], args.episodes, args.seed)
        save_results(name, scores, args.seed, args.output)
        print(f"Overall average: {scores.mean():.1f} steps")
    print(f"\nResults saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
