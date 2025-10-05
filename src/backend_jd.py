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
import requests

from fastapi import FastAPI
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
    min_no_cv_you_want:int
    min_no_days_you_want_to_collect_cv:int
    Cv_requirement:Annotated[str,Field(description="check enough cv or not")]
    Cv_history:Annotated[list[str],operator.add]
    full_cv:Annotated[list[str],operator.add]
    retry_cv:int
    max_retry_cv:int
    selected_student_for_interview:Annotated[list[dict],operator.add]
    #human_permission_for_interview:Annotated[str,Field(description="please select the interview date")]
    interview_date:Annotated[str,Field(description="here we select the interview date for student")]
    interview_time:Annotated[str,Field(description="here we select the interview time for student")]
    mail_generated_for_selected_students:Annotated[list[dict],operator.add]
    #mails_sent:Annotated[list[str],operator.add]

#Jd store here 
jd=[]

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
    try:
        response=generator_llm.invoke(message).content
    except Exception as e:
        raise Exception("LLM call fail")

    return {"tweet":response,"tweet_history":[response]}


# define the evaluation node
structured_evaluator_llm = evaluator_llm.with_structured_output(output_schema)

@traceable(name="evaluate_Jd", tags=["dimension:Analysis"], metadata={"dimension": "Analysis the tweet"})
def jd_evaluation(state:Jd)->Jd:
    query=f"Evaluate this job discription {state['tweet']} for this topic {state['topic']} and give a feedback "
    try:
        response=structured_evaluator_llm.invoke(query)
    except Exception as e:
        raise Exception("run time error ")
        

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
        jd.append(state["tweet"])
        return 'approved'
    else:
        return 'needs_improvement'
    

#----------------------Post Jd in Linkdin--------------------------------------------------------------------

@traceable(name="Post JD on LinkedIn", tags=["dimension:Post"], metadata={"dimension": "post job on LinkedIn"})
def post_linkedin(state: Jd) -> Jd:
    """
    Post the job description to LinkedIn via your FastAPI service.
    Expects `state['tweet']` to contain the JD.
    """

    # Load API URL and token from environment
    api_url = os.getenv("LINKEDIN_API_URL")  # e.g., "http://51.21.221.225:8000/post-job"
    access_token = os.getenv("LINKEDIN_ACCESS_TOKEN")  # your token in .env

    if not api_url:
        raise ValueError("LINKEDIN_API_URL not set in environment variables!")
    if not access_token:
        raise ValueError("LINKEDIN_ACCESS_TOKEN not set in environment variables!")

    # Only post the job text (no form, no image)
    data = {
        "access_token": access_token,
        "job_text": state["tweet"]
    }

    response = requests.post(api_url, data=data)

    if response.status_code != 200:
        raise Exception(f"Failed to post job: {response.status_code} - {response.text}")

    print("✅ Job posted on LinkedIn successfully!")
    return {"linkedin_post_status": "success", "linkedin_response": response.json()}



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

    if num_pdfs < state["min_no_cv_you_want"]:
        print(f"Less than {state["min_no_cv_you_want"]} resumes found.So we  Waiting for {wait} seconds again ...")
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


#------------------------Nodes for collect CV and then store the metadata of CV into database and summary into state---------------
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


#--------------------------------------------Human permission for date and time----------------------------------------
def fix_date_time(state:Jd):
    interview_date=state["interview_date"]
    interview_time=state["interview_time"]

    return {"interview_date":interview_date,"interview_time":interview_time}



# -----------------------------Here our LLM generate a email for selected student----------------------------------------

# Pydantic for mail 
class PydanticMail(BaseModel):
    mail: Annotated[str, Field(description="Generate the mail with date also and date should given date")]
    date: Annotated[str, Field(description="Here give the date and time  you selected for interview")]

    


    
#define the mail generator 
mail_llm=generator_llm.with_structured_output(PydanticMail)

def mail_generated_llm(state:Jd):
    mail_history=[]
    n=len(state["selected_student_for_interview"])
    a=0
    for i in state["selected_student_for_interview"]:
        Query=f""" generated a mail for this student name {i["name"]}
    and also give  a interview data {state["interview_date"]} and time should be  {state["interview_time"]} with addition time 
     {a*30} min  in the mail
    """ 
        response=mail_llm.invoke(Query)
        mail_history.append({"mail":response.mail,"date":response.date})
        a+=1
    return {"mail_generated_for_selected_students":mail_history}


#------------------------mail sending tool-----------------------------------------------------------------------

# @tool
# def send_email_tool(to_email: str, subject: str, body: str):
#     """Send an email to the candidate using environment variables for credentials."""
    
#     sender_email = os.getenv("EMAIL_USER")
#     sender_password = os.getenv("EMAIL_PASSWORD")
    
#     if not sender_email or not sender_password:
#         raise ValueError("Email credentials not set in environment variables!")
    
#     # Construct email message
#     msg = MIMEText(body, "plain")
#     msg["Subject"] = subject
#     msg["From"] = sender_email
#     msg["To"] = to_email
    
#     # Send mail using SMTP SSL
#     with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
#         server.login(sender_email, sender_password)
#         server.sendmail(sender_email, to_email, msg.as_string())
    
#     return f"✅ Email sent to {to_email}"

# def send_mails_node(state: Jd):
#     sent_history = []
#     for student, mail_data in zip(state["selected_student_for_interview"],
#                                   state["mail_generated_for_selected_students"]):
#         subject = "Interview Invitation"
#         body = mail_data["mail"]
#         candidate_email = student["email"]  # make sure your state has emails!

#         result = send_email_tool.func(candidate_email, subject, body)  # call tool
#         sent_history.append({"student": student["name"], "status": result})
    
#     return {"mails_sent": sent_history}


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
graph.add_node("fix_date_time",fix_date_time)
#graph.add_node("send_mails_node",send_mails_node)


#add edges 
graph.add_edge(START,"jd_genearation")
graph.add_edge("jd_genearation","jd_evaluation")

#add conditional edge
graph.add_conditional_edges("jd_evaluation", route_evaluation, {'approved':'check_cvs' , 'needs_improvement': 'optimize_tweet'})
graph.add_edge("optimize_tweet","jd_evaluation")
graph.add_conditional_edges("check_cvs",conditional_cv,{'enough_resumes':'summarize_cv','needs_more_resumes':"check_cvs","stop_checking":'summarize_cv'})
graph.add_edge("summarize_cv","embedding_cv")
graph.add_edge("embedding_cv","fix_date_time")
graph.add_edge("fix_date_time","mail_generated_llm")
graph.add_edge("mail_generated_llm",END)
#graph.add_edge("send_mails_node",END)


workflow = graph.compile()



# #define initial state
# initial_state = {
#     "topic": "generate Job description for my company name Laxmi chect fund ,For this topic Data science ,with required skill,python,Mlops,ML,DL",
#     "iteration": 0,
#     "max_iteration": 5,
#     "retry_cv":0,
#     "max_retry_cv":3,
#     "min_no_cv_you_want":1

# }
# result = workflow.invoke(initial_state)


# # print(result["feedback"])
# # # print(result["tweet"])
# # print("selected students:",result['selected_student_for_interview'])
# # print("mail_histroy",result["mail_generated_for_selected_students"])


# #------------------------add API--------------------------------------------------------------------------------
# from fastapi import FastAPI
# app=FastAPI()

# @app.post("/predict")
# def comlplete_workflow():
#     initial_state = {
#     "topic": "generate Job description for my company name Laxmi chect fund ,For this topic Data science ,with required skill,python,Mlops,ML,DL",
#     "iteration": 0,
#     "max_iteration": 5,
#     "retry_cv":0,
#     "max_retry_cv":3,
#     "min_no_cv_you_want":1
#     }
#     result = workflow.invoke(initial_state)
#     return {"result":result}

