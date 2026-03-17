# SAMA-RL: A Self-adapting Memetic Algorithm with Reinforcement Learning for Multi-objective Aircraft Recovery

This repository provides the official implementation of the paper:

**"A Self-adapting Memetic Algorithm with Reinforcement Learning for Multi-objective Aircraft Recovery Problems"**

---

## 📂 Repository Structure

The core implementation consists of the following modules:

- `Algo_Main.py`  
  Entry point of the algorithm. It defines the problem instance and runs the optimization process.

- `SAMA_RL.py`  
  Main algorithm framework based on a multi-objective evolutionary algorithm (Geatpy-based).  
  Includes:
  - population evolution
  - non-dominated sorting
  - reinforcement learning-based operator selection
  - archive update and solution management

- `Operators.py`  
  Problem-specific encoding, initialization, and evaluation functions.  
  Includes:
  - solution encoding (flight-circle representation)
  - initialization strategies
  - airport capacity modeling
  - constraint evaluation (maintenance, slot capacity, etc.)

- `Local_Search.py`  
  Local search and repair operators.  
  Includes:
  - flight insertion (`Here_Insert`) :contentReference[oaicite:0]{index=0}
  - flight exchange (`Here_Cross`)
  - cancellation and repair strategies

---

## ⚙️ Requirements

- Python 3.8+
- NumPy
- Pandas
- Geatpy

Install dependencies:

```bash
pip install numpy pandas geatpy
