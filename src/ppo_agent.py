"""
ppo_agent.py
============
Proximal Policy Optimization (PPO) for topic sequencing in the Dewey Tutorial Agent.

We implement PPO with:
  - A two-layer softmax actor policy (linear in numpy)
  - A value-function critic baseline to reduce variance
  - Clipped surrogate objective: L^CLIP = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
  - Generalized Advantage Estimation (GAE) for advantage computation

State:  [mastery x8, recent_perf x8, fatigue, questions/100]  -> dim=18
Action: topic index (0-7)                                       -> n_actions=8
"""

import numpy as np
from typing import List, Tuple, Dict
from environment import TOPICS


STATE_DIM = 18   # 8 mastery + 8 recent_perf + fatigue + q_count
N_ACTIONS = len(TOPICS)


class LinearNetwork:
    """
    Simple two-layer network (numpy only): Linear -> ReLU -> Linear.
    Supports forward pass and gradient computation via finite differences.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        # He initialization
        self.W1 = rng.standard_normal((input_dim, hidden_dim)) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.standard_normal((hidden_dim, output_dim)) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (output, hidden) for use in gradient computation."""
        h = np.maximum(0, x @ self.W1 + self.b1)   # ReLU
        out = h @ self.W2 + self.b2
        return out, h

    def get_params(self) -> List[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]

    def set_params(self, params: List[np.ndarray]):
        self.W1, self.b1, self.W2, self.b2 = params

    def num_params(self) -> int:
        return sum(p.size for p in self.get_params())


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


class PPOAgent:
    """
    PPO agent for selecting which topic to teach next.

    Uses clipped surrogate loss with GAE-lambda advantage estimation.
    Policy and value share a hidden layer (actor-critic architecture).
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        n_actions: int = N_ACTIONS,
        hidden_dim: int = 64,
        lr_actor: float = 3e-3,
        lr_critic: float = 5e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        n_epochs: int = 4,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.n_epochs = n_epochs
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        # Actor and Critic networks
        self.actor = LinearNetwork(state_dim, hidden_dim, n_actions, seed=42)
        self.critic = LinearNetwork(state_dim, hidden_dim, 1, seed=43)

        # Rollout buffer
        self.buffer_states: List[np.ndarray] = []
        self.buffer_actions: List[int] = []
        self.buffer_rewards: List[float] = []
        self.buffer_log_probs: List[float] = []
        self.buffer_values: List[float] = []
        self.buffer_dones: List[bool] = []

        # Training metrics
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []
        self.entropies: List[float] = []
        self.episode_rewards: List[float] = []

    def get_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """
        Sample action from policy. Returns (action, log_prob, value).
        """
        logits, _ = self.actor.forward(state)
        probs = softmax(logits)
        # Epsilon-greedy exploration blend (decreases over time)
        action = np.random.choice(self.n_actions, p=probs)
        log_prob = float(np.log(probs[action] + 1e-8))

        value_out, _ = self.critic.forward(state)
        value = float(value_out[0])

        return action, log_prob, value

    def store_transition(self, state, action, reward, log_prob, value, done):
        self.buffer_states.append(state.copy())
        self.buffer_actions.append(action)
        self.buffer_rewards.append(reward)
        self.buffer_log_probs.append(log_prob)
        self.buffer_values.append(value)
        self.buffer_dones.append(done)

    def compute_gae(self, next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generalized Advantage Estimation."""
        values = np.array(self.buffer_values + [next_value])
        rewards = np.array(self.buffer_rewards)
        dones = np.array(self.buffer_dones, dtype=float)

        advantages = np.zeros(len(rewards))
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + np.array(self.buffer_values)
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def _policy_grad_update(self, states, actions, old_log_probs, advantages):
        """
        PPO clipped policy gradient update via finite-difference gradient.
        We compute gradient of clipped surrogate objective.
        """
        total_loss = 0.0
        grad_W1 = np.zeros_like(self.actor.W1)
        grad_b1 = np.zeros_like(self.actor.b1)
        grad_W2 = np.zeros_like(self.actor.W2)
        grad_b2 = np.zeros_like(self.actor.b2)

        for state, action, old_lp, adv in zip(states, actions, old_log_probs, advantages):
            logits, h = self.actor.forward(state)
            probs = softmax(logits)
            new_lp = float(np.log(probs[action] + 1e-8))

            ratio = np.exp(new_lp - old_lp)
            surr1 = ratio * adv
            surr2 = np.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
            loss = -min(surr1, surr2)
            total_loss += loss

            # Entropy bonus
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            loss -= self.entropy_coef * entropy

            # Policy gradient: d/d_logits of log_prob * advantage (clipped)
            # d log_prob / d logits = one_hot(action) - probs
            use_clipped = float(ratio) < (1 - self.clip_eps) or float(ratio) > (1 + self.clip_eps)
            if use_clipped and float(ratio) * adv < float(np.clip(ratio, 1-self.clip_eps, 1+self.clip_eps)) * adv:
                grad_scale = 0.0  # gradient zero when clipped and not beneficial
            else:
                grad_scale = adv / (abs(adv) + 1e-8) * min(abs(ratio), 1 + self.clip_eps)

            d_logits = probs.copy()
            d_logits[action] -= 1.0
            d_logits = -d_logits * grad_scale  # negative because we minimize loss

            # Entropy gradient
            d_logits += self.entropy_coef * (np.log(probs + 1e-8) + 1 - np.sum(probs * (np.log(probs + 1e-8) + 1)))

            grad_W2 += np.outer(h, d_logits)
            grad_b2 += d_logits
            d_h = d_logits @ self.actor.W2.T
            d_h *= (h > 0).astype(float)  # ReLU backprop
            grad_W1 += np.outer(state, d_h)
            grad_b1 += d_h

        n = len(states)
        self.actor.W1 -= self.lr_actor * grad_W1 / n
        self.actor.b1 -= self.lr_actor * grad_b1 / n
        self.actor.W2 -= self.lr_actor * grad_W2 / n
        self.actor.b2 -= self.lr_actor * grad_b2 / n

        return total_loss / n

    def _value_update(self, states, returns):
        """MSE value function update."""
        total_loss = 0.0
        grad_W1 = np.zeros_like(self.critic.W1)
        grad_b1 = np.zeros_like(self.critic.b1)
        grad_W2 = np.zeros_like(self.critic.W2)
        grad_b2 = np.zeros_like(self.critic.b2)

        for state, ret in zip(states, returns):
            val_out, h = self.critic.forward(state)
            val = val_out[0]
            error = val - ret
            total_loss += 0.5 * error ** 2

            d_out = np.array([error])
            grad_W2 += np.outer(h, d_out)
            grad_b2 += d_out
            d_h = d_out @ self.critic.W2.T
            d_h *= (h > 0).astype(float)
            grad_W1 += np.outer(state, d_h)
            grad_b1 += d_h

        n = len(states)
        self.critic.W1 -= self.lr_critic * grad_W1 / n
        self.critic.b1 -= self.lr_critic * grad_b1 / n
        self.critic.W2 -= self.lr_critic * grad_W2 / n
        self.critic.b2 -= self.lr_critic * grad_b2 / n
        return total_loss / n

    def update(self, next_state: np.ndarray, done: bool):
        """Run PPO update on collected rollout."""
        if len(self.buffer_states) < 2:
            return

        next_val_out, _ = self.critic.forward(next_state)
        next_value = 0.0 if done else float(next_val_out[0])

        advantages, returns = self.compute_gae(next_value)
        states = np.array(self.buffer_states)
        actions = np.array(self.buffer_actions)
        old_log_probs = np.array(self.buffer_log_probs)

        ep_reward = sum(self.buffer_rewards)
        self.episode_rewards.append(ep_reward)

        for _ in range(self.n_epochs):
            # Shuffle mini-batch
            idx = np.random.permutation(len(states))
            p_loss = self._policy_grad_update(
                states[idx], actions[idx], old_log_probs[idx], advantages[idx]
            )
            v_loss = self._value_update(states[idx], returns[idx])
            self.policy_losses.append(p_loss)
            self.value_losses.append(v_loss)

        self._clear_buffer()

    def _clear_buffer(self):
        self.buffer_states.clear()
        self.buffer_actions.clear()
        self.buffer_rewards.clear()
        self.buffer_log_probs.clear()
        self.buffer_values.clear()
        self.buffer_dones.clear()

    def get_greedy_action(self, state: np.ndarray) -> int:
        """Return the greedy (highest probability) action."""
        logits, _ = self.actor.forward(state)
        return int(np.argmax(softmax(logits)))
