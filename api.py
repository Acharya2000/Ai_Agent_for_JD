from fastapi import FastAPI
from pydantic import BaseModel,Field
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.backend_jd import workflow
from typing import Annotated
app = FastAPI()

class WorkflowInput(BaseModel):
    api_key: str
    topic: str
    iteration: int = 0
    max_iteration: int = 5
    retry_cv: int = 0
    max_retry_cv: int = 3
    min_no_cv_you_want: int = 1
    interview_date: str
    interview_time: str



@app.post("/predict")
def complete_workflow(input_data: WorkflowInput):
    # Set the API key dynamically
    # Prepare your workflow initial_state
    initial_state = input_data.dict()

    # Call your LangGraph workflow
    result = workflow.invoke(initial_state)

    return {"result": result}
