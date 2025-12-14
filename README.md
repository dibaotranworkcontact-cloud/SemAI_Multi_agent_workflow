# SemAI - Multi-Agent Derivative Pricing Workflow

A sophisticated multi-agent AI system for derivative pricing using the CrewAI framework. This project implements a **two-crew sequential architecture**: the **Computational Crew** (8 agents) develops and trains models, followed by the **Validation Crew** (5 agents) that validates, replicates, and ensures compliance.

---

## 🚀 Complete Workflow Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SEMAI WORKFLOW PIPELINE                           │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  STEP 1: DATASET SELECTION (Human Supervisor)                       │  │
│   │    • Yahoo Finance  • Alpha Vantage  • CSV Import  • Market API     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│   ╔═════════════════════════════════════════════════════════════════════╗  │
│   ║              CREW 1: COMPUTATIONAL CREW (8 Agents)                  ║  │
│   ╠═════════════════════════════════════════════════════════════════════╣  │
│   ║                                                                     ║  │
│   ║  ┌─────────────── LEVEL 1: DATA PROCESSING ───────────────────┐    ║  │
│   ║  │                                                             │    ║  │
│   ║  │  [1] Data Extraction ──► [2] EDA ──► [3] Feature Engineering│    ║  │
│   ║  │         │                                        │          │    ║  │
│   ║  │         ▼                                        ▼          │    ║  │
│   ║  │  [6] Model Training ◄── [5] Meta Tuning ◄── [4] Model Eval  │    ║  │
│   ║  │                                                             │    ║  │
│   ║  └─────────────────────────────────────────────────────────────┘    ║  │
│   ║                              │                                      ║  │
│   ║                              ▼                                      ║  │
│   ║  ┌─────────────── LEVEL 2: OVERSIGHT ─────────────────────────┐    ║  │
│   ║  │                                                             │    ║  │
│   ║  │       [7] Judge Agent ──────► [8] Documentation Writer      │    ║  │
│   ║  │      (Quality Audit)         (ComputationalCrewDocumentation)│    ║  │
│   ║  │                                                             │    ║  │
│   ║  └─────────────────────────────────────────────────────────────┘    ║  │
│   ║                                                                     ║  │
│   ╚═════════════════════════════════════════════════════════════════════╝  │
│                                    │                                        │
│                    OUTPUT: ComputationalCrewDocumentation                   │
│                                    │                                        │
│                                    ▼                                        │
│   ╔═════════════════════════════════════════════════════════════════════╗  │
│   ║               CREW 2: VALIDATION CREW (5 Agents)                    ║  │
│   ╠═════════════════════════════════════════════════════════════════════╣  │
│   ║                                                                     ║  │
│   ║  ┌─────────────── LEVEL 1: VALIDATION TESTING ────────────────┐    ║  │
│   ║  │                                                             │    ║  │
│   ║  │  [1] Documentation      [2] Model         [3] Robustness    │    ║  │
│   ║  │      Compliance ────►   Replication ────►     Check         │    ║  │
│   ║  │      Checker            Agent             Agent             │    ║  │
│   ║  │                                                             │    ║  │
│   ║  └─────────────────────────────────────────────────────────────┘    ║  │
│   ║                              │                                      ║  │
│   ║                              ▼                                      ║  │
│   ║  ┌─────────────── LEVEL 2: COMPLIANCE & DOCS ─────────────────┐    ║  │
│   ║  │                                                             │    ║  │
│   ║  │   [4] Compliance Judge ──────► [5] Validation Doc Writer    │    ║  │
│   ║  │   (Risk Assessment)           (ComprehensiveSummary)        │    ║  │
│   ║  │                                                             │    ║  │
│   ║  └─────────────────────────────────────────────────────────────┘    ║  │
│   ║                                                                     ║  │
│   ╚═════════════════════════════════════════════════════════════════════╝  │
│                                    │                                        │
│                       OUTPUT: ComprehensiveSummary                          │
│                                    │                                        │
│                                    ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  STEP 4: HUMAN SUPERVISOR DECISION                                  │  │
│   │    • FEEDBACK → Return to Crew 1 with modifications                 │  │
│   │    • CONTINUE → Next iteration with expanded dataset                │  │
│   │    • END → Finalize and deploy                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Computational Crew (8 Agents)

*Develops, trains, and documents ML models for derivative pricing*

| # | Agent | Role | Goal | LLM |
|---|-------|------|------|-----|
| 1 | **Data Extraction Agent** | Data Engineer | Extract data, remove duplicates & erroneous entries | GPT-3.5-Turbo |
| 2 | **EDA Agent** | Data Engineer | Split datasets, perform exploratory analysis, create validation sets | GPT-3.5-Turbo |
| 3 | **Feature Engineering Agent** | Data Engineer | Handle missing values, outliers, construct derived features | GPT-3.5-Turbo |
| 4 | **Model Evaluation Agent** | Senior Data Engineer | Learn model documentation, evaluate performance, create templates | GPT-4-Turbo |
| 5 | **Meta Tuning Agent** | Senior Data Analyst | Optimize hyperparameters using MSE, RMSE, MAE, R² metrics | GPT-4-Turbo |
| 6 | **Model Training Agent** | Data Engineer | Train models, test performance, generate alignment plots | GPT-3.5-Turbo |
| 7 | **Judge Agent** | Chief Financial Officer | Audit for data leakage, compliance, code accuracy | GPT-4-Turbo |
| 8 | **Documentation Writer** | Secretary | Create ComputationalCrewDocumentation | GPT-3.5-Turbo |

---

## ✅ Validation Crew (5 Agents)

*Validates outputs, ensures compliance, and produces final documentation*

| # | Agent | Role | Goal | LLM |
|---|-------|------|------|-----|
| 1 | **Documentation Compliance Checker** | Senior Data Analyst | Compare outputs against InstitutionLegalChecklist | DeepSeek-V3.1 |
| 2 | **Model Replication Agent** | Testing Engineer | Replicate training, verify reproducibility of metrics | GPT-3.5-Turbo |
| 3 | **Robustness Check Agent** | Testing Engineer | Test on drifted datasets, assess stability under distribution shift | GPT-3.5-Turbo |
| 4 | **Compliance Judge Agent** | Chief Risk Management Officer | Judge interpretability, legal compliance, conceptual soundness | GPT-4-Turbo |
| 5 | **Validation Documentation Writer** | Secretary | Create ComprehensiveSummary combining both crews | DeepSeek-V3.1 |

---

## 📋 Sequential Workflow Algorithm

```
START
│
├─► STEP 1: DATASET SELECTION (Human Supervisor)
│       ├── Option A: Yahoo Finance (SPY, option chains)
│       ├── Option B: Alpha Vantage (OHLCV, indicators)
│       ├── Option C: CSV Import (custom data)
│       └── Option D: Market Data API (real-time feeds)
│
├─► STEP 2: COMPUTATIONAL CREW EXECUTION
│       │
│       ├── L1-1: data_extraction_agent
│       │         → Clean data, remove duplicates/errors
│       │
│       ├── L1-2: eda_agent
│       │         → Split train/validation, create drifted datasets
│       │
│       ├── L1-3: feature_engineering_agent
│       │         → Handle missing values, engineer features
│       │
│       ├── L1-4: model_evaluation_agent
│       │         → Learn models, create coding templates
│       │
│       ├── L1-5: meta_tuning_agent
│       │         → Hyperparameter optimization (MSE, RMSE, MAE, R²)
│       │
│       ├── L1-6: model_training_agent
│       │         → Train best model, generate plots
│       │
│       ├── L2-7: judge_agent
│       │         → Audit for data leakage, compliance
│       │
│       └── L2-8: documentation_writer
│                 → OUTPUT: "ComputationalCrewDocumentation"
│
├─► STEP 3: VALIDATION CREW EXECUTION
│       │
│       ├── L1-1: documentation_compliance_checker
│       │         → Compare against InstitutionLegalChecklist
│       │
│       ├── L1-2: model_replication_agent
│       │         → Replicate training, verify metrics match
│       │
│       ├── L1-3: robustness_check_agent
│       │         → Test on drifted data, assess stability
│       │
│       ├── L2-4: compliance_judge_agent
│       │         → Final risk assessment & recommendations
│       │
│       └── L2-5: validation_documentation_writer
│                 → OUTPUT: "ComprehensiveSummary"
│
├─► STEP 4: HUMAN SUPERVISOR DECISION
│       ├── FEEDBACK: "Modify and re-run" → goto STEP 2
│       ├── CONTINUE: "Expand dataset, iterate" → goto STEP 1
│       └── END: "Approve for deployment" → FINALIZE
│
END → Final Outputs:
      • ComputationalCrewDocumentation
      • ComprehensiveSummary
      • Trained Model Artifacts
```

---

## ⚠️ Setup Required - API Keys

**Before running this project, you must configure your API keys:**

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` and add your API keys:**
   ```env
   OPENAI_API_KEY=sk-your-actual-openai-key
   TOGETHER_API_KEY=your-together-api-key
   SERPER_API_KEY=your-serper-key  # Optional
   ```

3. **Get your API keys from:**
   - OpenAI: https://platform.openai.com/api-keys
   - Together AI (for DeepSeek): https://api.together.xyz/settings/api-keys
   - Serper (optional): https://serper.dev/api-key

> ⚠️ **Never commit your `.env` file to version control!**

---

## 📁 Project Structure

```
semai/
├── README.md                    # This file
├── .env.example                 # Template for API keys (copy to .env)
├── ALGORITHM_SEMAI_CREW.md      # Workflow algorithm & human interaction
├── pyproject.toml               # Project dependencies
├── run.bat / run.ps1            # Quick start scripts
├── docs/                        # Documentation
│   ├── guides/                  # User guides (RAG, Tools, Data Extraction)
│   ├── models/                  # Model references (Neural Networks, SDE)
│   └── references/              # Technical references (Guardrails, Dataset)
├── src/semai/                   # Source code
│   ├── config/                  # Agent & task configurations (YAML)
│   ├── tools/                   # Custom CrewAI tools
│   └── data/                    # Sample datasets
└── tests/                       # Test files & examples
```

---

## ⚡ Quick Start

```bash
# 1. Install UV package manager
pip install uv

# 2. Install dependencies
crewai install

# 3. Configure your API keys (REQUIRED!)
cp .env.example .env
# Edit .env with your actual API keys

# 4. Run the crew
crewai run
```

---

## 🔧 Configuration

| File | Purpose |
|------|---------|
| `src/semai/config/agents.yaml` | Agent definitions (8 agents with roles, LLMs, tools) |
| `src/semai/config/tasks.yaml` | Task definitions and dependencies |
| `src/semai/crew.py` | Crew orchestration logic |
| `src/semai/agent_softmax_config.py` | Softmax metrics configuration |

---

## 📚 Documentation

| Category | Files |
|----------|-------|
| **Guides** | RAG System, Tools Overview, Hyperparameter Testing, Data Extraction |
| **Models** | Neural Network References, SDE Models |
| **References** | Guardrails, Dataset Schema, Available Tools |

---

## 🛠️ Tools Available

- **FileReadTool** / **FileWriterTool** / **DirectoryReadTool** - File operations
- **LoadCSVTool** - Load and parse CSV datasets
- **HyperparameterTestingTool** - Systematic hyperparameter optimization
- **CAGTool** - Corrective Augmented Generation
- **CodeInterpreterTool** - Execute Python code

---

## 📖 Support

- [CrewAI Documentation](https://docs.crewai.com)
- [CrewAI GitHub](https://github.com/joaomdmoura/crewai)
- [ALGORITHM_SEMAI_CREW.md](ALGORITHM_SEMAI_CREW.md) - Complete workflow details
