from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Annotated

class OutputStructure(BaseModel):
    name: Annotated[str, Field(description="Full name of the student")]
    phone: Annotated[str, Field(description="Phone number of the student")]
    email:Annotated[str,Field(description="Email address of the student ")]
    summary: Annotated[str, Field(description="Summary of the resume within 100 words")]

# Load API key
load_dotenv()

# LLM
model = ChatOpenAI(model="gpt-4o-mini")
structure_model = model.with_structured_output(OutputStructure)

# Load resume
loader = PyPDFLoader(r"D:\Ai_Agent_for_JD\Cv_folder\142402011_Suman_Acharya_core_cv.pdf")
docs = loader.load()
resume_text = " ".join([doc.page_content for doc in docs])

# Query
query = f"""
Extract the following information from this resume:
1. Full name of the student
2. Phone number (if available)
3. Email of the student
4. A summary of the resume (within 100 words)

Resume text:
{resume_text}
"""

# Structured response
response = structure_model.invoke(query)

print(response)
