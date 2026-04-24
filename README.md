# reinforcement-learning-dqn-reinforce

Implementation of three reinforcement learning algorithms trained on OpenAI Gym environments.

## Algorithms
- **Deep Q-Learning (DQN)** — value-based RL with experience replay and target network
- **REINFORCE** — policy gradient algorithm
- **REINFORCE with Baseline** — variance-reduced policy gradient using a critic network

## Project Structure
```
src/
├── deep_q.py                  # DQN model and loss
├── reinforce.py               # REINFORCE policy network
├── reinforce_with_baseline.py # Actor-critic implementation
├── train.py                   # Training loop
└── visual.py                  # Visualization utilities
assignment.py                  # Entry point
```

## Requirements
```bash
pip install tensorflow numpy gymnasium
```

## How to Run
```bash
# Train Deep Q-Learning
python assignment.py --model DEEP_Q --env CartPole-v1 --num-episodes 650

# Train REINFORCE
python assignment.py --model REINFORCE --env CartPole-v1 --num-episodes 650

# Train REINFORCE with Baseline
python assignment.py --model REINFORCE_BASELINE --env CartPole-v1 --num-episodes 650

# Train on other environments
python assignment.py --model DEEP_Q --env LunarLander-v2 --num-episodes 1650

# Watch trained agent perform
python assignment.py --model DEEP_Q --env CartPole-v1 --watch

# Custom learning rate
python assignment.py --model REINFORCE --num-episodes 650 --lr 0.002
```

## Available Environments
- CartPole-v1
- LunarLander-v3
- MountainCar-v0
- Acrobot-v1

## Results
- DQN achieves average reward > 110 on CartPole-v1