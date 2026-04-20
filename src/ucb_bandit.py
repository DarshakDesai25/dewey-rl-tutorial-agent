"""
ucb_bandit.py
=============
Upper Confidence Bound (UCB1) Contextual Bandit for adaptive difficulty selection.

For each topic, the bandit maintains separate reward estimates and visit counts,
allowing it to personalize difficulty independently per topic.

UCB formula: a* = argmax_a [ Q(a) + c * sqrt(ln(t) / N(a)) ]
  - Q(a): estimated reward (learning gain) for difficulty a
  - c:    exploration constant (default 1.414 = sqrt(2))
  - t:    total timesteps
  - N(a): number of times action a was taken
"""

import numpy as np
import math
from typing import List, Dict, Optional
from environment import TOPICS, DIFFICULTIES


class UCBDifficultyBandit:
    """
    Per-topic UCB1 bandit for selecting question difficulty.
    Reward signal = normalized learning gain from student response.
    """

    def __init__(self, exploration_constant: float = 1.414):
        self.c = exploration_constant
        self.difficulties = DIFFICULTIES
        self.n_arms = len(DIFFICULTIES)

        # Per-topic statistics: shape (n_topics, n_arms)
        self.counts: Dict[str, np.ndarray] = {
            t: np.zeros(self.n_arms) for t in TOPICS
        }
        self.rewards: Dict[str, np.ndarray] = {
            t: np.zeros(self.n_arms) for t in TOPICS
        }
        self.t: Dict[str, int] = {t: 0 for t in TOPICS}

        # History for analysis
        self.selection_history: Dict[str, List[float]] = {t: [] for t in TOPICS}
        self.reward_history: Dict[str, List[float]] = {t: [] for t in TOPICS}

    def select_difficulty(self, topic: str, student_mastery: Optional[float] = None) -> float:
        """
        Select difficulty for a given topic using UCB1.
        Falls back to mastery-guided exploration in early rounds.
        """
        self.t[topic] += 1
        t = self.t[topic]

        # Any unvisited arm: must explore it first
        unvisited = [i for i, c in enumerate(self.counts[topic]) if c == 0]
        if unvisited:
            chosen_idx = unvisited[0]
        else:
            ucb_values = (
                self.rewards[topic] / (self.counts[topic] + 1e-9)
                + self.c * np.sqrt(np.log(t) / (self.counts[topic] + 1e-9))
            )
            chosen_idx = int(np.argmax(ucb_values))

        chosen_diff = self.difficulties[chosen_idx]
        self.selection_history[topic].append(chosen_diff)
        return chosen_diff

    def update(self, topic: str, difficulty: float, reward: float):
        """Update bandit statistics after observing reward."""
        idx = self.difficulties.index(difficulty)
        self.counts[topic][idx] += 1
        # Incremental mean update
        n = self.counts[topic][idx]
        self.rewards[topic][idx] += (reward - self.rewards[topic][idx]) / n
        self.reward_history[topic].append(reward)

    def get_best_difficulty(self, topic: str) -> float:
        """Return the currently estimated best difficulty for a topic."""
        if self.counts[topic].sum() == 0:
            return 0.4
        avg_rewards = self.rewards[topic] / (self.counts[topic] + 1e-9)
        return self.difficulties[int(np.argmax(avg_rewards))]

    def get_stats(self) -> Dict:
        """Return summary statistics for analysis."""
        stats = {}
        for t in TOPICS:
            avg = self.rewards[t] / (self.counts[t] + 1e-9)
            stats[t] = {
                "best_difficulty": self.get_best_difficulty(t),
                "counts": self.counts[t].tolist(),
                "avg_rewards": avg.tolist(),
            }
        return stats
