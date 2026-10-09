
# Maskable PPO: Invalid Action Masking in Reinforcement Learning

## Project Overview

This project investigates the use of Maskable Proximal Policy Optimization (Maskable PPO) for reinforcement learning in environments with invalid actions.

We study how action masking affects agent training and performance by comparing standard PPO with Maskable PPO.

## Selected Environment

**Gymnasium Taxi-v4**

The agent controls a taxi on a 5x5 grid. Its objective is to pick up a passenger and deliver them to the correct destination.

- Observation space: `Discrete(500)`
- Action space: `Discrete(6)`
- Rewards: -1 per step, +20 for successful delivery, -10 for illegal pickup/dropoff
- Episode termination: successful delivery
- Episode truncation: maximum step limit

Taxi-v4 provides state-dependent action masks, making it suitable for studying invalid action masking.

## Environment Compatibility

The environment has been tested with Gymnasium and sb3-contrib.

- ActionMasker wrapper: working
- Action mask retrieval: working
- Mask shape: `(6,)`
- Compatibility check: passed

## Project Structure

```text
notebooks/
    01_environment_setup.ipynb
src/
results/
README.md
requirements.txt
```

Additional notebooks and results will be added as the project progresses.

## Installation

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Open `notebooks/01_environment_setup.ipynb` in VS Code or Jupyter and execute the cells.

## Project Status

- [x] Environment selection and setup
- [x] Initial action mask verification
- [x] Maskable PPO interface compatibility check
- [ ] Action masking implementation
- [ ] PPO and Maskable PPO training
- [ ] Performance evaluation and comparison
