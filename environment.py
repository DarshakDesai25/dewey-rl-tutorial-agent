"""
environment.py
==============
Simulated Student Environment for the Dewey RL Tutorial Agent.

The student model is grounded in Vygotsky's Zone of Proximal Development (ZPD):
learning is maximized when difficulty is slightly above current mastery.
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


TOPICS = [
    "variables_and_types",
    "control_flow",
    "functions",
    "recursion",
    "data_structures",
    "algorithms",
    "object_oriented",
    "error_handling",
]

DIFFICULTIES = [0.2, 0.4, 0.6, 0.8, 1.0]
DIFFICULTY_LABELS = {0.2: "Beginner", 0.4: "Easy", 0.6: "Medium", 0.8: "Hard", 1.0: "Expert"}


@dataclass
class SimulatedStudent:
    """Represents a simulated student with latent mastery per topic."""
    learning_rate: float = 0.12
    initial_mastery: float = 0.15
    fatigue_rate: float = 0.02
    recover_rate: float = 0.1
    p_guess: float = 0.15      # probability of guessing correctly
    p_slip: float = 0.08       # probability of slipping despite knowing

    # Runtime state
    mastery: Dict[str, float] = field(default_factory=dict)
    fatigue: float = 0.0
    response_history: Dict[str, List[bool]] = field(default_factory=dict)
    questions_answered: int = 0

    def __post_init__(self):
        for topic in TOPICS:
            self.mastery[topic] = self.initial_mastery + np.random.uniform(-0.05, 0.05)
            self.response_history[topic] = []

    def answer_question(self, topic: str, difficulty: float) -> dict:
        """
        Simulate student answering a question.
        Learning follows Vygotsky's ZPD: gain is maximized when difficulty
        is just above current mastery (~0.1 higher). Poor calibration causes
        large learning loss (sharp Gaussian falloff around ZPD center).
        """
        mastery = self.mastery[topic]
        fatigue_penalty = self.fatigue * 0.25

        # ZPD center = mastery + 0.1; Gaussian falloff, sigma=0.18
        zpd_center = min(mastery + 0.1, 0.95)
        zpd_alignment = float(np.exp(-((difficulty - zpd_center) ** 2) / (2 * 0.18 ** 2)))

        # P(correct)
        p_correct_base = mastery * (1.0 - self.p_slip) + (1.0 - mastery) * self.p_guess
        p_correct = float(np.clip(p_correct_base - fatigue_penalty, 0.05, 0.97))
        correct = random.random() < p_correct

        # Learning gain: heavily ZPD-gated so difficulty calibration matters
        max_gain = self.learning_rate * (1.0 - mastery)
        gain = max_gain * zpd_alignment * (1.0 if correct else 0.35)
        gain = max(gain, 0.001)

        pre_mastery = self.mastery[topic]
        self.mastery[topic] = float(np.clip(mastery + gain, 0.0, 1.0))
        actual_gain = self.mastery[topic] - pre_mastery

        self.fatigue = float(np.clip(self.fatigue + self.fatigue_rate - 0.004, 0.0, 1.0))
        self.response_history[topic].append(correct)
        self.questions_answered += 1

        return {
            "correct":        correct,
            "topic":          topic,
            "difficulty":     difficulty,
            "pre_mastery":    pre_mastery,
            "post_mastery":   self.mastery[topic],
            "learning_gain":  actual_gain,
            "zpd_alignment":  zpd_alignment,
            "zpd_center":     zpd_center,
            "fatigue":        self.fatigue,
            "p_correct":      p_correct,
        }

    def overall_mastery(self) -> float:
        return float(np.mean(list(self.mastery.values())))

    def get_state_vector(self) -> np.ndarray:
        """Return a normalized state vector for the RL agent."""
        mastery_vec = [self.mastery[t] for t in TOPICS]
        recent_perf = []
        for t in TOPICS:
            hist = self.response_history[t][-3:]
            recent_perf.append(np.mean(hist) if hist else 0.5)
        state = mastery_vec + recent_perf + [self.fatigue, self.questions_answered / 100.0]
        return np.array(state, dtype=np.float32)

    def reset(self):
        """Reset student to initial state (new session)."""
        self.__post_init__()

# Alias for backward compatibility
StudentProfile = SimulatedStudent

