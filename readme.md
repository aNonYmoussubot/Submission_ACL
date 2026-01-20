# TrustTable: A Neuro-Symbolic Auditing Framework for Faithful Table QA

**TrustTable** is a neuro-symbolic framework designed to audit the faithfulness of Large Language Models (LLMs) in Table Question Answering (TableQA). It addresses the "Right Answer, Wrong Reason" problem (process hallucinations) by enforcing strict factual grounding and logical validity through a dual-path verification mechanism.

This repository contains the official implementation of the paper **"TrustTable: A Neuro-Symbolic Auditing Framework for Faithful Table QA"**.

## Framework Architecture

TrustTable operates on a **"Generate-Audit-Refine"** paradigm, decoupling verification into orthogonal dimensions:

1. **FactChecker (Grounding)**:
   - Synthesizes and executes **Pandas** code to verify if the retrieved data (entities, values) actually exists in the table.
   - Intercepts **Type 3 Errors** (Hallucinations).
2. **LogicAuditor (Reasoning)**:
   - Auto-formalizes natural language reasoning into **Z3 SMT** constraints to verify mathematical and logical entailment.
   - Intercepts **Type 2 Errors** (Spurious Logic).
3. **Consistency Monitor**:
   - Checks if the executed reasoning result matches the final textual answer.
   - Intercepts **Type 4 Errors** (Execution Inconsistency).
4. **Label-Free Refinement**:
   - A feedback loop that uses symbolic error reports (e.g., "Counter-example found") to guide the LLM to self-correct without requiring gold labels.

------

## Setup & Installation

### Prerequisites

- Python 3.9 or higher
- An OpenAI-compatible API Key (e.g., DeepSeek, OpenAI)

### Installation

1. Clone the repository:

   

2. Install dependencies:

   

   *(Core dependencies include: `pandas`, `z3-solver`, `numpy`, `openai`)*

------

## Configuration

Before running the code, you need to configure the LLM provider.

Open `configs/config.py` and update the settings:



------

##  Usage

To run the full evaluation pipeline on the provided dataset sample:



The script will:

1. Load the dataset (configured in `main.py`).
2. Decompose reasoning traces into atomic steps.
3. Run the **Neuro-Symbolic Verification Pipeline** (FactChecker + Z3Auditor).
4. Trigger **Refinement** for rejected samples.
5. Output a statistical report including metrics like **VCAR** (Verified Correct Answer Rate) and **CSR** (Correction Success Rate).

------

## Dataset: TrustTable-Bench

We provide the generation scripts for **TrustTable-Bench**, a diagnostic dataset constructed to test four reasoning topologies:

- **Type 1**: Faithful ($Z^+ A^+$)
- **Type 2**: Spurious ($Z^- A^+$) - *Targeting "Right Answer, Wrong Reason"*
- **Type 3**: Wrong ($Z^- A^-$)
- **Type 4**: Inconsistent ($Z^+ A^-$)

You can find the generation logic in `raw_datasets/generate_rational_data_wtq_full.py`.

------

