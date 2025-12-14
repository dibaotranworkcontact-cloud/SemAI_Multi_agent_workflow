# CrewAI Tools Overview

## CrewAI Built-in Tools

| Category | Tool | Purpose |
|----------|------|---------|
| **File & Directory** | FileReadTool | Read files from disk |
| | FileWriteTool | Write and save files |
| | DirectoryReadTool | List and navigate directories |
| **Web & Search** | BrowserTool | Browse and interact with websites |
| | SerperDevTool | Google search via SerperDev API |
| | WebsiteSearchTool | Search within specific websites |
| | ScrapeWebsiteTool | Extract content from web pages |
| **Code & Development** | CodeInterpreterTool | Execute Python code |
| | CodeDocsSearchTool | Search code documentation |
| | GithubSearchTool | Search GitHub repositories |
| **Data Operations** | CSVSearchTool | Search and analyze CSV data |
| | PDFSearchTool | Extract and search PDF documents |
| | JSONSearchTool | Query and parse JSON files |
| | TXTSearchTool | Search text file content |
| **System Integration** | BashTool | Execute shell/bash commands |
| | GitTool | Perform Git operations |
| | DatabaseTool | Query databases |
| **Specialized** | YahooFinanceTool | Retrieve financial data |
| | CalculatorTool | Perform calculations |
| | XMLSearchTool | Search XML files |

## Custom Tools for Your Project

| Tool | Purpose | Key Features |
|------|---------|--------------|
| **HyperparameterTestingTool** | Optimize ML model hyperparameters | Random search, grid search, multiple metrics (MSAE, R², RMSE, MAE), JSON persistence, visualization |
| **Task Feedback Handler** | Manage task feedback and quality validation | Feedback collection, output validation, corrective guidance, history tracking, iterative improvement |
| **Data Input Handler** | Validate and preprocess data | Data validation, format normalization, anomaly detection, multi-source support (CSV, APIs, databases) |
| **API Handler** | Facilitate external API communication | Authentication management, request formatting, rate limiting, response parsing, error handling with retries |

## Tool Categories and Use Cases

| Use Case | Recommended Tools |
|----------|-------------------|
| **Data Science Workflow** | FileReadTool, CSVSearchTool, CodeInterpreterTool, FileWriteTool, Data Input Handler |
| **ML Model Optimization** | HyperparameterTestingTool, CodeInterpreterTool, FileWriteTool, Task Feedback Handler |
| **Financial Analysis** | API Handler (Alpha Vantage), YahooFinanceTool, CSVSearchTool, CalculatorTool |
| **Web Research** | BrowserTool, SerperDevTool, WebsiteSearchTool, ScrapeWebsiteTool |
| **Data Engineering** | BashTool, FileReadTool, DatabaseTool, GitTool, Data Input Handler |
| **Documentation Writing** | FileReadTool, FileWriteTool, DirectoryReadTool, CSVSearchTool |
| **Full ML Pipeline** | Data Input Handler, HyperparameterTestingTool, API Handler, Task Feedback Handler, CodeInterpreterTool |

## Integration Architecture

```
Agent → Tool Selection → Tool Execution → Result Processing → Feedback
   ↓
Multiple agents can share tools or have specialized tool sets
```

CrewAI's built-in tools provide general-purpose functionality for common operations, while custom tools (HyperparameterTestingTool, Task Feedback Handler, Data Input Handler, API Handler) are optimized for derivative pricing and machine learning workflows. Together, they enable agents to complete complex tasks from data preparation through model optimization to final analysis.

## Quick Integration Example

```python
from crewai import Agent
from semai.tools import (
    HyperparameterTestingTool,
    TaskFeedbackHandler,
    DataInputHandler,
    APIHandler
)
from crewai_tools import FileReadTool, CodeInterpreterTool

# Create specialized agent with combined tools
ml_agent = Agent(
    role="ML Engineer",
    goal="Optimize derivative pricing models",
    tools=[
        DataInputHandler(),           # Prepare data
        HyperparameterTestingTool(),  # Optimize parameters
        CodeInterpreterTool(),        # Execute code
        TaskFeedbackHandler(),        # Validate results
        FileReadTool()                # Access files
    ]
)
```

