# import all necessary module 

from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated,Literal
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import operator
import os
from langchain_core.messages import HumanMessage,SystemMessage
import time
from langchain_community.document_loaders import PyPDFLoader



#set up the openAi model 
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

#this llm genrate the post 
generator_llm=ChatOpenAI(api_key=api_key,model="gpt-4o")
#this llm evaluate the post
evaluator_llm=ChatOpenAI(api_key=api_key,model="gpt-4o")
#this llm update the post
optimizer_llm=ChatOpenAI(api_key=api_key,model="gpt-4o")
#this llm give mobile no,email,and resume summary
resume_llm=ChatOpenAI(model="gpt-4o-mini")


#define the state
class Jd(TypedDict):
    topic : Annotated[str,Field(description="here we give the topic of JD")]
    tweet: Annotated[str,Field(description="Here llm gives us the JD")]
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str
    iteration: int
    max_iteration: Annotated[int,Field(description="max no iteration ")]
    tweet_history: Annotated[list[str], operator.add]
    feedback_history: Annotated[list[str], operator.add]
    Cv_requirement:Annotated[str,Field(description="check enough cv or not")]
    Cv_history:Annotated[list[str],operator.add]
    retry_cv:int
    max_retry_cv:int



#pydantic schema for the output of evaluation node
class output_schema(BaseModel):
    evaluation:Literal["approved", "needs_improvement"]= Field(..., description="Final evaluation result.")

    feedback:Annotated[str,Field(..., description="feedback for the tweet.")]


#pydantatic schema for resume 
class OutputStructure(BaseModel):
    name: Annotated[str, Field(description="Full name of the student")]
    phone: Annotated[str, Field(description="Phone number of the student")]
    email:Annotated[str,Field(description="Email address of the student ")]
    summary: Annotated[str, Field(description="Summary of the resume within 100 words")]

resume_output_llm=resume_llm.with_structured_output(OutputStructure)

#define the jd_generation node
def jd_genearation(state:Jd)->Jd:
    message=[
        SystemMessage(content="you are a post genrator for a particular job topic"),
        HumanMessage(content=f"generate a job description on this topic {state['topic']}")
    ]
    response=generator_llm.invoke(message).content

    return {"tweet":response,"tweet_history":[response]}


# define the evaluation node
structured_evaluator_llm = evaluator_llm.with_structured_output(output_schema)

def jd_evaluation(state:Jd)->Jd:
    query=f"Evaluate this job discription {state['tweet']} for this topic {state['topic']} and give a feedback "
    response=structured_evaluator_llm.invoke(query)


    return {"evaluation":response.evaluation,"feedback":response.feedback,"feedback_history":[response.feedback]}
    


# define jd_update node
def optimize_tweet(state:Jd):

    messages = [
        SystemMessage(content="You punch up tweets for virality and humor based on given feedback."),
        HumanMessage(content=f"""
Improve the tweet based on this feedback:
"{state['feedback']}"

Topic: "{state['topic']}"
Original Tweet:
{state['tweet']}

Re-write it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters.
""")
    ]

    response = optimizer_llm.invoke(messages).content
    iteration = state['iteration'] + 1

    return {'tweet': response, 'iteration': iteration, 'tweet_history': [response]}



# Conditional node for JD update 
def route_evaluation(state:Jd):

    if state['evaluation'] == 'approved' or state['iteration'] >= state['max_iteration']:
        return 'approved'
    else:
        return 'needs_improvement'

# cv check node 
def check_cvs(state: Jd) -> Jd:
    folder_path = "Cv_folder"  # your CV folder
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    #num_pdfs = len(pdf_files)

    # print(f"Found {num_pdfs} resumes")
    #waiting for some time 
    wait=60
    print(f"waiting for {wait} seconds")
    time.sleep(wait) 

    #check  no of CV after waiting for some 
    num_pdfs = len(pdf_files)
    print(f"Found {num_pdfs} resumes")


    retry_cv=state["retry_cv"]+1

    if num_pdfs < 2:
        print(f"Less than 5 resumes found.So we  Waiting for {wait} seconds again ...")
        return {"Cv_requirement": "needs_more_resumes","retry_cv":retry_cv}  # temporary signal
    else:
        return {"Cv_requirement": "enough_resumes","retry_cv":0}
    

# conditional node to check no of CV enough or not 
def conditional_cv(state:Jd)->Jd:
    if state["Cv_requirement"]=="needs_more_resumes" and  state["retry_cv"]<state["max_retry_cv"]:
        return "needs_more_resumes"
    elif state["Cv_requirement"]=="enough_resumes":
        return "enough_resumes"
    else:
        return "stop_checking"


# Nodes for collect CV and then store the metadata of CV into datbase and summary into state
import sqlite3

def summarize_cv(state: Jd) -> Jd:
    folder_path = "Cv_folder"
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pdf")]
    summary_history = []
    
    # --- Setup SQLite connection ---
    conn = sqlite3.connect("resumes.db")
    cursor = conn.cursor()
    
    # Create table if not exists
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS candidates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        phone TEXT,
        email TEXT UNIQUE,   -- make email unique
        summary TEXT
    )
    """)
    
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        docs = loader.load()
        text = " ".join([doc.page_content for doc in docs])
        
        query = f"""
Extract the following information from this resume:
1. Full name of the student
2. Phone number (if available)
3. Email of the student
4. A summary of the resume (within 100 words)

Resume text:
{text}
"""
        response = resume_output_llm.invoke(query)

        # Save summary in state
        summary_history.append(response.summary)

        # --- Check if email already exists ---
        cursor.execute("SELECT id FROM candidates WHERE email = ?", (response.email,))
        existing = cursor.fetchone()
        
        if not existing:  # only insert if not found
            cursor.execute("""
            INSERT INTO candidates (name, phone, email, summary) VALUES (?, ?, ?, ?)
            """, (response.name, response.phone, response.email, response.summary))
    
    # Commit and close DB connection
    conn.commit()
    conn.close()

    return {"Cv_history": summary_history}

    


 

#create the graph 
graph=StateGraph(Jd)
# add nodes 
graph.add_node("jd_genearation",jd_genearation)
graph.add_node("jd_evaluation",jd_evaluation)
graph.add_node("optimize_tweet",optimize_tweet)
graph.add_node('check_cvs',check_cvs)
graph.add_node('summarize_cv',summarize_cv)

#add edges 
graph.add_edge(START,"jd_genearation")
graph.add_edge("jd_genearation","jd_evaluation")

#add conditional edge
graph.add_conditional_edges("jd_evaluation", route_evaluation, {'approved':'check_cvs' , 'needs_improvement': 'optimize_tweet'})
graph.add_edge("optimize_tweet","jd_evaluation")
graph.add_conditional_edges("check_cvs",conditional_cv,{'enough_resumes':'summarize_cv','needs_more_resumes':"check_cvs","stop_checking":'summarize_cv'})
graph.add_edge("summarize_cv",END)



workflow = graph.compile()

workflow


#define initial state
initial_state = {
    "topic": "Data Science",
    "iteration": 0,
    "max_iteration": 5,
    "retry_cv":0,
    "max_retry_cv":3

}
result = workflow.invoke(initial_state)


# print(result["feedback"])
# print(result["tweet"])
print(result['Cv_history'])