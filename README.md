# 🧬 LLM-Driven Autonomous MLOps: RNA 3D Structure Prediction

![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue?style=for-the-badge&logo=kaggle)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![AI Automated](https://img.shields.io/badge/AI-Automated_MLOps-purple?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

## 📌 Overview & Repository Purpose

This repository showcases an experimental, fully automated AI-driven framework built to tackle the **[Stanford RNA 3D Folding 2](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2)** Kaggle competition. 

Rather than relying on manual feature engineering, heuristic tuning, or traditional human-in-the-loop ML practices, this project deployed an **"Autonomous MLOps Pipeline"**. An ensemble of large language model (LLM) agents operated continuously to research structural biology, inject physics-informed constraints into PyTorch models, validate execution locally, automatically push kernels to Kaggle via its API, and dynamically learn from Leaderboard feedback over the course of **300+ autonomous iterations**.

## 📊 Competition Results & Performance

Through relentless automated trial and error, the agentic loop successfully adapted domain-specific structural constraints, achieving the following results on the public leaderboard:

* **Top Score:** `0.41124`
* **1st Place Score:** `0.55488`
* **Score Difference to 1st:** `-0.14364`
* **Final Rank:** `905` out of `1877` participating teams
* **Percentile:** **Top 48.2%**

*(Note: In this competition, a higher score denotes greater topological accuracy and structural alignment with the ground truth).*

Given that the entire pipeline—from literature research to code injection and Kaggle submission—was driven completely autonomously by an LLM without human structural biologists, reaching the top 48% demonstrates the viability of agentic workflows in highly specialized scientific domains.

## 🏗️ System Architecture: The Agentic Loop

The core engine (`scripts/autonomous_mlops_pipeline.sh`) orchestrated several distinct AI "personas", each responsible for a specific segment of the MLOps lifecycle:

1. **Scout (Researcher):** Explored the problem space by fetching structural RNA invariants. For example, it identified that the A-form helical symmetry requires a strict 5.95Å distance between consecutive C1' atoms, and that the minor groove width should be maintained around 13.5Å.
2. **Coder (Constraint Injector):** Utilized tools like `constraint_injector.py` to dynamically parse and modify base inference scripts. It algorithmically injected geometric constraints (e.g., O2'(i) to O2'(i+1) distance adjustments) using NumPy/PyTorch vector math.
3. **Tester (QA & Validator):** Executed the newly patched code in an isolated local environment against dummy validation sets. If syntax errors or shape mismatches occurred, it triggered an immediate self-correction loop via `code_patcher.py` before burning Kaggle submission limits.
4. **Orchestrator (Deployment):** Communicated with the Kaggle API to package notebooks, manage metadata (`kernel-metadata.json`), and monitor remote execution status in real-time.
5. **Reflector (Meta-Learner):** Downloaded historical leaderboard scores and parsed remote Kaggle logs. It summarized failures into `docs/learnings.md` to ensure the system never made the same programmatic mistake twice (e.g., switching from hardcoded paths to `Path().rglob()` to survive Kaggle's hidden test sets).

## 🧐 Critical Analysis & Retrospective

While the autonomous loop was a remarkable engineering feat, pushing the boundaries of what zero-human-intervention MLOps can achieve, it also exposed profound limitations in current LLM reasoning paradigms:

### ✅ Key Triumphs
* **Zero-Shot Domain Adaptation:** The LLMs successfully extracted highly specific, niche biophysics principles without human prompting. Implementing a "harmonic tether" to prevent local shrinkage while maintaining global stability was a highly sophisticated computational biology technique derived purely by the agent.
* **Resilient Infrastructure Debugging:** The agent demonstrated impressive resilience against infrastructure bugs. When Kaggle's hidden evaluation environments threw `FileNotFoundError`s due to dynamic file structures, the Reflector agent diagnosed the stack trace, rewrote the data loader using recursive pathing, and successfully deployed the fix (Iteration v61).

### ⚠️ Limitations & The "Frankenstein" Problem
* **The Trap of Local Optimization:** The most glaring limitation was the LLM's tendency to obsess over *local* constraints at the expense of *global* topology. The Coder agent would perfectly optimize inter-strand phosphorus distances or individual ribose puckers, but lacked the spatial awareness to realize that these local adjustments were cumulatively twisting the RNA backbone into physically impossible "Frankenstein" structures.
* **High Remote Attrition Rate:** Over 80% of the 300+ submissions resulted in Kaggle `SubmissionStatus.ERROR`. LLMs struggled to anticipate "invisible" constraints such as Out-Of-Memory (OOM) errors on Kaggle's P100 GPUs or strict 9-hour execution timeouts, which could not be perfectly simulated in the local Tester environment.
* **Asynchronous Bottlenecks:** The pipeline's speed was severely bottlenecked by Kaggle's remote evaluation time. Because the Reflector needed the final score to update its strategy, the loop spent hours idle, highlighting the need for stronger local surrogate models.

## 📂 Repository Structure

The workspace has been refactored and sanitized for portfolio presentation:

```text
.
├── src/                # Core reusable python modules, physics math, and model architectures
├── scripts/            # Automation scripts: autonomous_mlops_pipeline.sh, constraint_injector.py
├── logs/               # System logs, watcher outputs, and historical diff analyses
├── docs/               # System prompts, competitor context, strategy text, and meta-learnings
├── archive/            # Historical candidate scripts (v1 to v349) documenting the evolution
├── data/               # Raw and processed structural datasets (ignored by git)
├── submissions/        # Generated CSV outputs for Kaggle evaluation
├── tests/              # Local QA scripts, geometric calculators, and torsion tests
├── requirements.txt    # Python dependency matrix
└── README.md           # Project documentation
```

## 🚀 Setup & Reproduction

1. **Environment Setup:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Authentication:** 
   Ensure `~/.kaggle/kaggle.json` or `docs/api_keys.txt` is correctly configured with your Kaggle API credentials.
3. **Execute the Autonomous Pipeline:**
   ```bash
   bash scripts/autonomous_mlops_pipeline.sh
   ```
   *(Note: As the competition has officially concluded, running this pipeline will perform local geometric patching and validation, but remote submissions will no longer update the active leaderboard).*

---
*This repository stands as an empirical case study in the future of Autonomous Data Science—highlighting the immense power of agentic MLOps, while underscoring the indispensable need for human-guided architectural vision when dealing with complex, multi-dimensional physical constraints.*
