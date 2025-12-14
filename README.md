# SemAI - Multi-Agent Derivative Pricing Workflow

A sophisticated multi-agent AI system for derivative pricing using the CrewAI framework. This project implements a two-level hierarchical crew architecture with 8 specialized agents working together to extract data, engineer features, tune hyperparameters, train models, and produce comprehensive documentation.

---

## 🚀 Workflow Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SEMAI DERIVATIVE PRICING CREW                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     LEVEL 1: DATA PROCESSING                        │   │
│  │                                                                     │   │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐ │   │
│  │   │    Data      │───►│     EDA      │───►│     Feature          │ │   │
│  │   │  Extraction  │    │    Agent     │    │   Engineering        │ │   │
│  │   │    Agent     │    │              │    │      Agent           │ │   │
│  │   └──────────────┘    └──────────────┘    └──────────────────────┘ │   │
│  │          │                                           │              │   │
│  │          ▼                                           ▼              │   │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐ │   │
│  │   │    Model     │◄───│    Meta      │◄───│       Model          │ │   │
│  │   │   Training   │    │   Tuning     │    │     Evaluation       │ │   │
│  │   │    Agent     │    │    Agent     │    │       Agent          │ │   │
│  │   └──────────────┘    └──────────────┘    └──────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   LEVEL 2: OVERSIGHT & DOCUMENTATION                │   │
│  │                                                                     │   │
│  │   ┌──────────────────────────┐    ┌──────────────────────────────┐ │   │
│  │   │       Judge Agent        │    │    Documentation Writer      │ │   │
│  │   │  (Quality & Compliance)  │    │   (Technical Documentation)  │ │   │
│  │   └──────────────────────────┘    └──────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│                         ┌──────────────────┐                               │
│                         │  Human Supervisor │                               │
│                         │   FEEDBACK/END    │                               │
│                         └──────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Agent Crew

| Agent | Role | Goal | LLM |
|-------|------|------|-----|
| **Data Extraction Agent** | Data Engineer | Extract data, remove duplicates & erroneous entries | GPT-3.5-Turbo |
| **EDA Agent** | Data Engineer | Split datasets, perform exploratory analysis, create validation sets | GPT-3.5-Turbo |
| **Feature Engineering Agent** | Data Engineer | Handle missing values, outliers, construct derived features | GPT-3.5-Turbo |
| **Model Evaluation Agent** | Senior Data Engineer | Learn model documentation, evaluate performance, create templates | GPT-4-Turbo |
| **Meta Tuning Agent** | Senior Data Analyst | Optimize hyperparameters using MSE, RMSE, MAE, R² metrics | GPT-4-Turbo |
| **Model Training Agent** | Data Engineer | Train models, test performance, generate alignment plots | GPT-3.5-Turbo |
| **Judge Agent** | Chief Financial Officer | Audit for data leakage, compliance, code accuracy | GPT-4-Turbo |
| **Documentation Writer** | Secretary | Create comprehensive technical documentation | GPT-3.5-Turbo |

---

## 📋 Brief Workflow Algorithm

```
START
│
├─► 1. DATASET SELECTION (Human chooses)
│       ├── Option A: Yahoo Finance (SPY, option chains)
│       ├── Option B: Alpha Vantage (OHLCV, indicators)
│       ├── Option C: CSV Import (custom data)
│       └── Option D: Market Data API (real-time feeds)
│
├─► 2. EXECUTE CREW
│       ├── Level-1: data_extraction → eda → feature_engineering
│       │            → model_evaluation → meta_tuning → model_training
│       └── Level-2: judge_agent (assess) + documentation_writer (record)
│
├─► 3. HUMAN DECISION
│       ├── FEEDBACK: "Re-run with modifications" → goto 2
│       ├── CONTINUE: "Proceed to next iteration" → goto 2
│       └── END: "Finalize documentation" → END
│
END → Output: ComputationalCrewDocumentation
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
