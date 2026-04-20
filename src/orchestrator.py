"""
orchestrator.py
===============
Dewey Tutorial Agent Orchestrator

Integrates PPO (topic sequencing), UCB Bandit (difficulty), and BKT tool
into a unified agentic tutoring system. Three agents share the same
persistent student for a fair apples-to-apples comparison.

Experimental Design:
  - ONE student per agent, shared across all episodes (no full reset)
  - Episodes = tutoring sessions; mastery accumulates over time
  - We measure HOW QUICKLY each agent brings the student to proficiency
  - PPO+UCB wins by finding optimal difficulty per ZPD, not just lucky random choices
"""

import numpy as np
import random
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from environment import SimulatedStudent, TOPICS, DIFFICULTIES
from ppo_agent import PPOAgent
from ucb_bandit import UCBDifficultyBandit
from tools import BayesianKnowledgeTracingTool


@dataclass
class SessionLog:
    episode: int
    interactions: List[Dict] = field(default_factory=list)
    final_mastery: Dict[str, float] = field(default_factory=dict)
    total_reward: float = 0.0
    n_questions: int = 0


def compute_reward(response: dict, bkt_mastery: float) -> float:
    """
    Multi-objective reward:
      60%  learning gain  (primary: maximize knowledge acquisition)
      20%  BKT signal     (calibrated mastery improvement)
      20%  efficiency     (penalize fatigue waste)
    """
    learning_gain  = np.clip(response["learning_gain"] * 8.0, 0, 1)
    mastery_signal = np.clip(bkt_mastery * response["learning_gain"] * 5.0, 0, 1)
    efficiency     = np.clip(1.0 - response["fatigue"], 0, 1)
    return float(0.60 * learning_gain + 0.20 * mastery_signal + 0.20 * efficiency)


class DeweyTutorialOrchestrator:
    """PPO + UCB + BKT integrated tutorial agent."""

    def __init__(self, questions_per_episode: int = 25, update_every: int = 10, verbose: bool = False):
        self.questions_per_episode = questions_per_episode
        self.update_every = update_every
        self.verbose = verbose

        self.ppo_agent  = PPOAgent()
        self.ucb_bandit = UCBDifficultyBandit(exploration_constant=1.414)
        self.bkt_tool   = BayesianKnowledgeTracingTool()

        self.session_logs:       List[SessionLog] = []
        self.mastery_over_time:  List[float]      = []
        self.reward_over_time:   List[float]      = []

    def run_episode(self, episode: int, student: SimulatedStudent) -> SessionLog:
        log = SessionLog(episode=episode)
        # Reset only fatigue between sessions (mastery persists — student remembers)
        student.fatigue = 0.0
        student.questions_answered = 0

        for step in range(self.questions_per_episode):
            bkt_mastery = self.bkt_tool.estimate_all_topics(student.response_history)
            state = student.get_state_vector()

            action, log_prob, value = self.ppo_agent.get_action(state)
            topic = TOPICS[action]

            difficulty = self.ucb_bandit.select_difficulty(topic, bkt_mastery[topic])
            response   = student.answer_question(topic, difficulty)
            bkt_after  = self.bkt_tool.estimate_mastery(topic, student.response_history[topic])
            reward     = compute_reward(response, bkt_after)

            self.ucb_bandit.update(topic, difficulty, response["learning_gain"] * 10.0)

            done = (step == self.questions_per_episode - 1)
            self.ppo_agent.store_transition(state, action, reward, log_prob, value, done)

            if (step + 1) % self.update_every == 0 or done:
                self.ppo_agent.update(student.get_state_vector(), done)

            log.interactions.append({
                "step": step, "topic": topic, "difficulty": difficulty,
                "correct": response["correct"], "learning_gain": response["learning_gain"],
                "reward": reward, "bkt_mastery": bkt_after,
            })
            log.total_reward += reward

        log.final_mastery = {t: student.mastery[t] for t in TOPICS}
        log.n_questions   = self.questions_per_episode

        self.session_logs.append(log)
        self.mastery_over_time.append(student.overall_mastery())
        self.reward_over_time.append(log.total_reward)

        if self.verbose and episode % 20 == 0:
            print(f"  Episode {episode:3d} | Mastery: {student.overall_mastery():.3f} | Reward: {log.total_reward:.3f}")

        return log

    def train(self, n_episodes: int) -> List[SessionLog]:
        student = SimulatedStudent(learning_rate=0.08, initial_mastery=0.12)
        print(f"🎓 Training Dewey PPO+UCB Agent for {n_episodes} episodes...")
        for ep in range(n_episodes):
            self.run_episode(ep, student)
        print(f"✅ Done. Final mastery: {student.overall_mastery():.3f}")
        return self.session_logs


# ─── Baseline Agents ──────────────────────────────────────────────────────────

class RandomBaselineAgent:
    """Randomly selects topic and difficulty — lower bound baseline."""
    def __init__(self, questions_per_episode: int = 25):
        self.qpe = questions_per_episode
        self.mastery_over_time: List[float] = []

    def train(self, n_episodes: int) -> None:
        print(f"🎲 Training Random Baseline for {n_episodes} episodes...")
        student = SimulatedStudent(learning_rate=0.08, initial_mastery=0.12)
        for _ in range(n_episodes):
            student.fatigue = 0.0
            student.questions_answered = 0
            for _ in range(self.qpe):
                student.answer_question(random.choice(TOPICS), random.choice(DIFFICULTIES))
            self.mastery_over_time.append(student.overall_mastery())
        print(f"✅ Random done. Final mastery: {np.mean(self.mastery_over_time[-10:]):.3f}")


class FixedDifficultyAgent:
    """BKT topic selection + fixed medium difficulty — ablation baseline."""
    def __init__(self, questions_per_episode: int = 25, difficulty: float = 0.4):
        self.qpe = questions_per_episode
        self.difficulty = difficulty
        self.mastery_over_time: List[float] = []
        self._bkt = BayesianKnowledgeTracingTool()

    def train(self, n_episodes: int) -> None:
        print(f"📏 Training Fixed-Difficulty Baseline for {n_episodes} episodes...")
        student = SimulatedStudent(learning_rate=0.08, initial_mastery=0.12)
        for _ in range(n_episodes):
            student.fatigue = 0.0
            student.questions_answered = 0
            for _ in range(self.qpe):
                mastery = self._bkt.estimate_all_topics(student.response_history)
                topic   = self._bkt.recommend_next_topic(mastery)
                student.answer_question(topic, self.difficulty)
            self.mastery_over_time.append(student.overall_mastery())
        print(f"✅ Fixed-diff done. Final mastery: {np.mean(self.mastery_over_time[-10:]):.3f}")
