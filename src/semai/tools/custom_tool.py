from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import pandas as pd
import os


class LoadCSVInput(BaseModel):
    """Input schema for LoadCSV tool."""
    file_path: str = Field(..., description="Full file path to the CSV file to load")


class LoadCSVTool(BaseTool):
    """Tool for loading CSV files from local paths."""
    name: str = "Load CSV File"
    description: str = (
        "Loads a CSV file from a local file path and returns basic statistics. "
        "Use the dataset_url parameter from the task to load the dataset."
    )
    args_schema: Type[BaseModel] = LoadCSVInput

    def _run(self, file_path: str) -> str:
        """Load CSV from local file path."""
        try:
            # Handle directory paths - find CSV in directory
            if os.path.isdir(file_path):
                csv_files = [f for f in os.listdir(file_path) if f.endswith('.csv')]
                if not csv_files:
                    return f"Error: No CSV files found in {file_path}"
                file_path = os.path.join(file_path, csv_files[0])
                print(f"Loading CSV: {csv_files[0]}")
            
            # Load the CSV
            df = pd.read_csv(file_path)
            
            summary = f"""CSV loaded successfully from: {file_path}
Shape: {df.shape[0]} rows x {df.shape[1]} columns
Columns: {list(df.columns)}
Data types:
{df.dtypes.to_string()}
Missing values:
{df.isnull().sum().to_string()}
Basic statistics:
{df.describe().to_string()}"""
            
            return summary
        
        except Exception as e:
            return f"Error loading CSV: {str(e)}"


class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""
    argument: str = Field(..., description="Description of the argument.")

class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = (
        "Clear description for what this tool is useful for, your agent will need this information to use it."
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, argument: str) -> str:
        # Implementation goes here
        return "this is an example of a tool output, ignore it and move along."
