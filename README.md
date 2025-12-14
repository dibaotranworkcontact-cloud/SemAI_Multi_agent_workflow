# Semai Crew - Derivative Pricing Pipeline

A multi-agent AI system for derivative pricing using CrewAI framework.

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

## Project Structure

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
│   ├── config/                  # Agent & task configurations
│   ├── tools/                   # Custom CrewAI tools
│   └── data/                    # Sample datasets
└── tests/                       # Test files & examples
```

## Quick Start

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

## Agent Architecture

**Level-1 (Processing):** Data Extraction → Data Generator → Feature Engineering → Model Constructor → Meta Tuning → Model Training

**Level-2 (Oversight):** Performance Judge + Documentation Writer

See [ALGORITHM_SEMAI_CREW.md](ALGORITHM_SEMAI_CREW.md) for complete workflow.

## Configuration

- `src/semai/config/agents.yaml` - Agent definitions (8 agents with roles, LLMs, tools)
- `src/semai/config/tasks.yaml` - Task definitions
- `src/semai/crew.py` - Crew orchestration logic

## Documentation

| Category | Files |
|----------|-------|
| **Guides** | RAG System, Tools Overview, Hyperparameter Testing, Data Extraction |
| **Models** | Neural Network References, SDE Models |
| **References** | Guardrails, Dataset Schema, Available Tools |

## Support

- [CrewAI Documentation](https://docs.crewai.com)
- [CrewAI GitHub](https://github.com/joaomdmoura/crewai)
