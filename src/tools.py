"""
tools.py
========
Custom Tool: Bayesian Knowledge Tracing (BKT) Estimator

BKT is a well-validated algorithm from educational data mining (Corbett & Anderson, 1994).
It uses a Hidden Markov Model to estimate the probability that a student has learned
a concept, given their observable response history.

HMM Parameters:
  p_learn (T): Probability of transitioning from unlearned -> learned
  p_guess (G): Probability of correct response given unlearned state
  p_slip  (S): Probability of incorrect response given learned state
  p_init  (L0): Prior probability of knowing the skill initially
"""

from typing import List, Dict
from environment import TOPICS


class BayesianKnowledgeTracingTool:
    """
    Implements the BKT forward algorithm to estimate latent mastery.

    This tool is called by the TutorialAgentOrchestrator to get a
    calibrated mastery estimate that the RL agent uses as part of its state.
    Unlike raw accuracy, BKT accounts for guessing and slipping, providing
    a more robust signal for the policy.
    """

    # Default BKT parameters (calibrated on CS education data)
    DEFAULT_PARAMS = {
        "variables_and_types":  {"p_init": 0.20, "p_learn": 0.35, "p_guess": 0.20, "p_slip": 0.08},
        "control_flow":         {"p_init": 0.15, "p_learn": 0.28, "p_guess": 0.18, "p_slip": 0.10},
        "functions":            {"p_init": 0.12, "p_learn": 0.25, "p_guess": 0.15, "p_slip": 0.10},
        "recursion":            {"p_init": 0.08, "p_learn": 0.18, "p_guess": 0.12, "p_slip": 0.08},
        "data_structures":      {"p_init": 0.10, "p_learn": 0.20, "p_guess": 0.15, "p_slip": 0.09},
        "algorithms":           {"p_init": 0.08, "p_learn": 0.15, "p_guess": 0.12, "p_slip": 0.07},
        "object_oriented":      {"p_init": 0.10, "p_learn": 0.22, "p_guess": 0.15, "p_slip": 0.09},
        "error_handling":       {"p_init": 0.15, "p_learn": 0.30, "p_guess": 0.20, "p_slip": 0.10},
    }

    def __init__(self):
        self.params = self.DEFAULT_PARAMS.copy()
        self._estimate_cache: Dict[str, float] = {}

    def estimate_mastery(self, topic: str, response_history: List[bool]) -> float:
        """
        Run the BKT forward algorithm over a student's response history.

        Returns P(learned | responses), the posterior probability that the
        student has mastered the topic given all observed answers.

        Time complexity: O(n) where n = len(response_history)
        """
        if not response_history:
            return self.params[topic]["p_init"]

        p = self.params[topic]
        p_known = p["p_init"]

        for correct in response_history:
            # E-step: Update belief given observation
            if correct:
                numerator = p_known * (1.0 - p["p_slip"])
                denominator = numerator + (1.0 - p_known) * p["p_guess"]
            else:
                numerator = p_known * p["p_slip"]
                denominator = numerator + (1.0 - p_known) * (1.0 - p["p_guess"])

            p_known = numerator / (denominator + 1e-10)

            # M-step: Apply learning transition
            p_known = p_known + (1.0 - p_known) * p["p_learn"]

        return float(min(max(p_known, 0.0), 1.0))

    def estimate_all_topics(self, response_histories: Dict[str, List[bool]]) -> Dict[str, float]:
        """Estimate mastery for all topics and return a calibrated state."""
        return {
            topic: self.estimate_mastery(topic, response_histories.get(topic, []))
            for topic in TOPICS
        }

    def recommend_next_topic(self, mastery_estimates: Dict[str, float]) -> str:
        """
        Heuristic recommendation: prioritize topics with mid-range mastery
        (most learnable, not too easy/hard). Used as baseline comparison.
        """
        scores = {
            t: mastery_estimates[t] * (1.0 - mastery_estimates[t]) * 4  # peaks at 0.5
            for t in TOPICS
        }
        return max(scores, key=scores.get)

    def mastery_summary(self, mastery_estimates: Dict[str, float]) -> str:
        """Return a human-readable mastery summary."""
        lines = ["📊 BKT Mastery Estimates:"]
        for topic, mastery in mastery_estimates.items():
            bar = "█" * int(mastery * 10) + "░" * (10 - int(mastery * 10))
            level = "Beginner" if mastery < 0.3 else "Developing" if mastery < 0.6 else "Proficient" if mastery < 0.85 else "Expert"
            lines.append(f"  {topic:20s} [{bar}] {mastery:.2f} ({level})")
        return "\n".join(lines)
