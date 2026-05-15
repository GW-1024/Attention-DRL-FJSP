# Attention-DRL-FJSP
End-to-End Smart Scheduling: Solving the Flexible Job-shop Scheduling Problem (FJSP) via Attention-based Deep Reinforcement Learning. This framework achieves millisecond-level agile rescheduling and overcomes standard GPU memory bottlenecks through a novel large-to-small dynamic batch training strategy.

# Attention-DRL for FJSP

This repository contains the source code for solving the Flexible Job-shop Scheduling Problem (FJSP) using Deep Reinforcement Learning.

## Features
* **Global Perception:** Uses Multi-Head Self-Attention instead of traditional GNNs.
* **Dynamic Masking:** Guarantees 100% physically valid schedules.
* **Dynamic Batch Strategy:** A large-to-small batch training method to overcome GPU memory limits and escape local optima.

## Requirements
* Python 3.8+
* PyTorch 2.x
* OpenAI Gym

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the training script:
3. 1.trian the data  `train_experiment_Attention.py`
4. 2.validate ‘validate_experiment_Attention.py’
