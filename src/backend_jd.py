# import all necessary module 

from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated,Literal
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from dotenv import load_dotenv
from pydantic import BaseModel, Field,field_validator
import operator
import os
from langchain_core.messages import HumanMessage,SystemMessage
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langsmith import traceable
from datetime import datetime
from langchain_core.tools import tool
import smtplib
from email.mime.text import MIMEText


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
#embedding model 
emb_model=OpenAIEmbeddings(model='text-embedding-3-small')


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
    full_cv:Annotated[list[str],operator.add]
    retry_cv:int
    max_retry_cv:int
    selected_student_for_interview:Annotated[list[dict],operator.add]
    mail_generated_for_selected_students:Annotated[list[dict],operator.add]
    mails_sent:Annotated[list[str],operator.add]



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
    full_cv:Annotated[str,Field(description="Give a clean  text for the Full CV which represent the student CV like score,Skill,Project ")]

resume_output_llm=resume_llm.with_structured_output(OutputStructure)

#define the jd_generation node
@traceable(name="Generate Jd", tags=["dimension:language"], metadata={"dimension": "language"})
def jd_genearation(state:Jd)->Jd:
    message=[
        SystemMessage(content="you are a post genrator for a particular job topic"),
        HumanMessage(content=f"generate a job description on this topic {state['topic']}")
    ]
    response=generator_llm.invoke(message).content

    return {"tweet":response,"tweet_history":[response]}


# define the evaluation node
structured_evaluator_llm = evaluator_llm.with_structured_output(output_schema)

@traceable(name="evaluate_Jd", tags=["dimension:Analysis"], metadata={"dimension": "Analysis the tweet"})
def jd_evaluation(state:Jd)->Jd:
    query=f"Evaluate this job discription {state['tweet']} for this topic {state['topic']} and give a feedback "
    response=structured_evaluator_llm.invoke(query)


    return {"evaluation":response.evaluation,"feedback":response.feedback,"feedback_history":[response.feedback]}
    


# define jd_update node

@traceable(name="Update Jd", tags=["dimension:optimize"], metadata={"dimension": "optimize the tweet "})
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



#--------------------------------------Conditional node for JD update --------------------------------------------------
@traceable(name="Conditional Node for Jd", tags=["dimension:decision"], metadata={"dimension": "decision to go back to Optimize or not"})
def route_evaluation(state:Jd):

    if state['evaluation'] == 'approved' or state['iteration'] >= state['max_iteration']:
        return 'approved'
    else:
        return 'needs_improvement'

#-------------------------------------------cv check node--------------------------------------------------------
@traceable(name="Check_no_of_cv", tags=["dimension:Count CV"], metadata={"dimension": "Count no of application"})
def check_cvs(state: Jd) -> Jd:
    folder_path = "Cv_folder"  # your CV folder

    #waiting for some time 
    wait=60
    print(f"waiting for {wait} seconds")
    time.sleep(wait) 


    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    # #num_pdfs = len(pdf_files)

    # # print(f"Found {num_pdfs} resumes")
    # #waiting for some time 
    # wait=60
    # print(f"waiting for {wait} seconds")
    # time.sleep(wait) 

    #check  no of CV after waiting for some 
    num_pdfs = len(pdf_files)
    print(f"Found {num_pdfs} resumes")


    retry_cv=state["retry_cv"]+1

    if num_pdfs < 1:
        print(f"Less than 5 resumes found.So we  Waiting for {wait} seconds again ...")
        return {"Cv_requirement": "needs_more_resumes","retry_cv":retry_cv}  # temporary signal
    else:
        return {"Cv_requirement": "enough_resumes","retry_cv":0}
    

#---------------------------conditional node to check no of CV enough or not---------------------------------------

@traceable(name="Check_enough_Cv_or_not", tags=["dimension:Enough Cv"], metadata={"dimension": "Enough Cv or not"})
def conditional_cv(state:Jd)->Jd:
    if state["Cv_requirement"]=="needs_more_resumes" and  state["retry_cv"]<state["max_retry_cv"]:
        return "needs_more_resumes"
    elif state["Cv_requirement"]=="enough_resumes":
        return "enough_resumes"
    else:
        return "stop_checking"


#------------------------Nodes for collect CV and then store the metadata of CV into datbase and summary into state---------------
import sqlite3
@traceable(name="Summarize Cv", tags=["dimension:CV extract"], metadata={"dimension": "We collect the CV text "})
def summarize_cv(state: Jd) -> Jd:
    folder_path = "Cv_folder"
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pdf")]
    summary_history = []
    full_cv=[]
    
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
        summary TEXT,
        full_cv TEXT
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
5.A clean text for the entire CV 

Resume text:
{text}
"""
        response = resume_output_llm.invoke(query)

        # Save summary in state
        summary_history.append(response.summary)
        full_cv.append(response.full_cv)

        # --- Check if email already exists ---
        cursor.execute("SELECT id FROM candidates WHERE email = ?", (response.email,))
        existing = cursor.fetchone()
        
        if not existing:  # only insert if not found
            cursor.execute("""
            INSERT INTO candidates (name, phone, email, summary,full_cv) VALUES (?, ?, ?, ?,?)
            """, (response.name, response.phone, response.email, response.summary,response.full_cv))
    
    # Commit and close DB connection
    conn.commit()
    conn.close()

    return {"Cv_history": summary_history,"full_cv":full_cv}

    

#--------------------------------------define embedding and retrival node--------------------------------------------- 

@traceable(name="Retrive CV ", tags=["dimension:Retrive CV"], metadata={"dimension": "Here we retrive Student for the Job"})
def embedding_cv(state: Jd) -> Jd:
    # --- Load candidates from DB ---
    conn = sqlite3.connect("resumes.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name, phone, email, summary,full_cv FROM candidates")
    rows = cursor.fetchall()
    conn.close()

    #check if any student apply or not 
    if not rows:
        print("No candidates in DB to index.")
        return {"selected_student_for_interview": []}

    # Convert DB rows → Documents with metadata
    docs = [
        Document(
            page_content=row[4],  # full cv
            metadata={
                "name": row[0],
                "phone": row[1],
                "email": row[2]
            }
        )
        for row in rows
    ]

    # Build FAISS index
    vs = FAISS.from_documents(docs, emb_model)
    vs.save_local("faiss_index")

    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    results = retriever.invoke(state["tweet"])  # query with JD/tweet

    # Extract metadata of top matches
    top_matches = [
        {
            "name": doc.metadata.get("name"),
            "email": doc.metadata.get("email"),
            "phone": doc.metadata.get("phone"),
            #"matched_summary": doc.page_content
        }
        for doc in results
    ]

    return {"selected_student_for_interview": top_matches}


# -----------------------------Here our LLM generate a email for selected student----------------------------------------

# Pydantic for mail 
class PydanticMail(BaseModel):
    mail: Annotated[str, Field(description="Generate the mail with date also and date should be between 7/01/2026 to 9/30/2026")]
    date: Annotated[str, Field(description="Here give the date you selected for interview")]

    # @field_validator("date")
    # def validate_date_range(cls, value):
    #     try:
    #         # Parse input date (assuming format YYYY-MM-DD)
    #         dt = datetime.strptime(value, "%Y-%m-%d")
    #     except ValueError:
    #         raise ValueError("Date must be in format YYYY-MM-DD")

    #     # Allowed interval: July 1, 2025 → Sept 30, 2025
    #     start = datetime(2026, 7, 1)
    #     end = datetime(2026, 9, 30)

    #     if not (start <= dt <= end):
    #         raise ValueError("Date must be between July 2025 and September 2025")

    #     return value
    
#define the generator 
mail_llm=generator_llm.with_structured_output(PydanticMail)

def mail_generated_llm(state:Jd):
    mail_history=[]
    for i in state["selected_student_for_interview"]:
        Query=f""" generated a mail for this student name {i["name"]}
    and also give  a interview data in the mail
    """ 
        response=mail_llm.invoke(Query)
        mail_history.append({"mail":response.mail,"date":response.date})
    return {"mail_generated_for_selected_students":mail_history}

#----------------------------------Human in the loop----------------------------------------------------------------
def human_approval_date(state:Jd):
    pass 


#------------------------mail sending tool-----------------------------------------------------------------------

@tool
def send_email_tool(to_email: str, subject: str, body: str):
    """Send an email to the candidate using environment variables for credentials."""
    
    sender_email = os.getenv("EMAIL_USER")
    sender_password = os.getenv("EMAIL_PASSWORD")
    
    if not sender_email or not sender_password:
        raise ValueError("Email credentials not set in environment variables!")
    
    # Construct email message
    msg = MIMEText(body, "plain")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = to_email
    
    # Send mail using SMTP SSL
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, msg.as_string())
    
    return f"✅ Email sent to {to_email}"

def send_mails_node(state: Jd):
    sent_history = []
    for student, mail_data in zip(state["selected_student_for_interview"],
                                  state["mail_generated_for_selected_students"]):
        subject = "Interview Invitation"
        body = mail_data["mail"]
        candidate_email = student["email"]  # make sure your state has emails!

        result = send_email_tool.func(candidate_email, subject, body)  # call tool
        sent_history.append({"student": student["name"], "status": result})
    
    return {"mails_sent": sent_history}


#-----------------------------------------------create the graph------------------------------------------------------
graph=StateGraph(Jd)
# add nodes 
graph.add_node("jd_genearation",jd_genearation)
graph.add_node("jd_evaluation",jd_evaluation)
graph.add_node("optimize_tweet",optimize_tweet)
graph.add_node('check_cvs',check_cvs)
graph.add_node('summarize_cv',summarize_cv)
graph.add_node("embedding_cv",embedding_cv)
graph.add_node("mail_generated_llm",mail_generated_llm)
#graph.add_node("send_mails_node",send_mails_node)


#add edges 
graph.add_edge(START,"jd_genearation")
graph.add_edge("jd_genearation","jd_evaluation")

#add conditional edge
graph.add_conditional_edges("jd_evaluation", route_evaluation, {'approved':'check_cvs' , 'needs_improvement': 'optimize_tweet'})
graph.add_edge("optimize_tweet","jd_evaluation")
graph.add_conditional_edges("check_cvs",conditional_cv,{'enough_resumes':'summarize_cv','needs_more_resumes':"check_cvs","stop_checking":'summarize_cv'})
graph.add_edge("summarize_cv","embedding_cv")
graph.add_edge("embedding_cv","mail_generated_llm")
graph.add_edge("mail_generated_llm",END)
#graph.add_edge("send_mails_node",END)


workflow = graph.compile()

workflow


#define initial state
initial_state = {
    "topic": "generate Job description for my company name Laxmi chect fund ,For this topic Data science ,with required skill,python,Mlops,ML,DL",
    "iteration": 0,
    "max_iteration": 5,
    "retry_cv":0,
    "max_retry_cv":3

}
result = workflow.invoke(initial_state)


# print(result["feedback"])
# print(result["tweet"])
print("selected students:",result['selected_student_for_interview'])
print("mail_histroy",result["mail_generated_for_selected_students"])