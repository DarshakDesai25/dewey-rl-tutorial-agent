# Dewey RL Tutorial Agent

Reinforcement Learning for Adaptive Personalized Tutoring in the Dewey Agentic Framework.

## Overview

This project integrates two RL approaches into a Dewey-style tutorial agent:

| RL Method | Component | Purpose |
|-----------|-----------|---------|
| **PPO** (Policy Gradient) | `ppo_agent.py` | Learn optimal *topic sequencing* policy |
| **UCB1 Bandit** (Exploration) | `ucb_bandit.py` | Adaptively select *difficulty* per topic |
| **BKT** (Custom Tool) | `tools.py` | Calibrated mastery estimation |

## Requirements

```bash
pip install numpy matplotlib scipy
```

No PyTorch required — PPO is implemented in pure NumPy.

## Quick Start

```bash
# Run full experiment suite (generates all plots + report)
python run_experiments.py

# Generate PDF report (requires experiment_results.json)
python generate_report.py
```

## File Structure

```
dewey_rl_tutor/
├── environment.py        # Simulated student (ZPD-based learning model)
├── ppo_agent.py          # PPO with GAE, implemented in NumPy
├── ucb_bandit.py         # UCB1 contextual bandit (per-topic difficulty)
├── tools.py              # BKT custom tool (Bayesian Knowledge Tracing)
├── orchestrator.py       # Dewey agent orchestrator + baselines
├── run_experiments.py    # Full experiment suite w/ visualizations
├── generate_report.py    # PDF technical report generator
└── results/
    ├── learning_curves.png
    ├── speed_to_proficiency.png
    ├── ucb_convergence.png
    ├── topic_mastery_heatmap.png
    ├── before_after_transcript.txt
    ├── experiment_results.json
    └── technical_report.pdf
```

## Key Results

| Agent | Final Mastery | Notes |
|-------|--------------|-------|
| **PPO + UCB (Ours)** | **1.000 ± 0.000** | Adapts difficulty to student ZPD |
| Fixed Difficulty + BKT | 0.744 ± 0.009 | **Ceiling effect** — stuck below 75% |
| Random Baseline | 1.000 ± 0.000 | Converges slowly, high variance |

**Key finding**: Fixed difficulty teaching causes a mastery *ceiling effect* at ~74%. As student mastery grows past the fixed difficulty level, the Zone of Proximal Development (ZPD) shifts upward, making further learning nearly impossible. PPO+UCB's adaptive difficulty selection naturally tracks the ZPD, enabling full mastery.

## RL Design Choices

### PPO — Topic Sequencing
- **State**: 18-dim vector [BKT mastery ×8, recent accuracy ×8, fatigue, session_progress]
- **Action**: Topic index (8 choices)
- **Reward**: 0.60 × learning_gain + 0.20 × BKT_signal + 0.20 × efficiency
- **Architecture**: 2-layer actor-critic (64 hidden units), implemented in NumPy

### UCB1 — Difficulty Selection
- Maintains **independent** bandit per topic (8 topics × 5 difficulty arms)
- Reward signal: learning_gain × 10
- Converges to optimal difficulty per topic through exploration

### BKT Tool — Mastery Estimation
- Implements Corbett & Anderson (1994) forward algorithm
- Topic-specific parameters: P(L₀), P(T), P(G), P(S)
- Provides denoised mastery estimate that accounts for guessing and slipping

## Theoretical Foundation

The student simulation is grounded in Vygotsky's Zone of Proximal Development:

```
gain(d, m) = lr × (1 - m) × exp(-(d - (m+0.1))² / (2 × 0.18²))
```

where `d` is difficulty, `m` is current mastery. Learning is maximized when difficulty is 0.1 above current mastery.
