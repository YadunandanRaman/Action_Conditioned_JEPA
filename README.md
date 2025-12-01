# Action-Conditioned JEPA: Emergence of World Models via Epistemic Curiosity

**Author:** R Yadunandan  
**Status:** Preprint / Research Implementation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![MiniGrid](https://img.shields.io/badge/Environment-MiniGrid-green)](https://minigrid.farama.org/)

This repository contains the official PyTorch implementation of the paper **"Action-Conditioned JEPA: Emergence of World Models via Epistemic Curiosity"**.

## 📖 Abstract
Current AI systems largely rely on passive next-token prediction on static datasets. We argue that true understanding requires **agency**—the ability to act, predict consequences, and update beliefs based on prediction errors.

This project implements a **Unified Agent Architecture** that integrates:
1.  **Transformer World Model (JEPA):** Predicts latent state dynamics instead of pixels using Self-Attention and Rotary Embeddings (RoPE).
2.  **Epistemic Curiosity:** Uses prediction error (ICM + RND) as an intrinsic reward signal to drive exploration.
3.  **Active Agency:** A PPO policy that utilizes the learned world model to navigate and solve tasks.

The agent is trained end-to-end on **MiniGrid**, demonstrating emergent causal understanding and directed exploration.

---

## 🚀 Key Features
* **Single-File Implementation:** The entire architecture (Model, Training, PPO, Analysis) is contained in `main.py` for maximum reproducibility.
* **Causal Reasoning:** The World Model learns to attend to the **Action** token when predicting the **Next State**, enabling counterfactual reasoning.
* **Latent Dynamics:** Predicts in a compact latent space rather than pixel space, making it computationally efficient (~15M parameters).
* **Dual Curiosity:** Combines Inverse Dynamics (ICM) for controllability and Random Network Distillation (RND) for novelty.

---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YadunandanRaman/Action_Conditioned_JEPA.git](https://github.com/YadunandanRaman/Action_Conditioned_JEPA.git)
    cd Action-Conditioned-JEPA
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🏃 Usage

To train the agent and generate all analysis figures (Latent Space, Attention Maps, etc.) in one go:

```bash
python main.py
