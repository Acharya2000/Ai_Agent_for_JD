# import all necessary module 

from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated,Literal
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import operator
import os
from langchain_core.messages import HumanMessage,SystemMessage


#set up the openAi model 
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

#this llm genrate the post 
generator_llm=ChatOpenAI(api_key=api_key,model="gpt-4o")
#this llm evaluate the post
evaluator_llm=ChatOpenAI(api_key=api_key,model="gpt-4o")
#this llm update the post
optimizer_llm=ChatOpenAI(api_key=api_key,model="gpt-4o")


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



#pydantic schema for the output of evaluation node
class output_schema(BaseModel):
    evaluation:Literal["approved", "needs_improvement"]= Field(..., description="Final evaluation result.")

    feedback:Annotated[str,Field(..., description="feedback for the tweet.")]



#define the jd_generation node
def jd_genearation(state:Jd)->Jd:
    message=[
        SystemMessage(content="you are a post genrator for a particular job topic"),
        HumanMessage(content=f"generate a job description on this topic {state["topic"]}")
    ]
    response=generator_llm.invoke(message).content

    return {"tweet":response,"tweet_history":[response]}


# define the evaluation node
structured_evaluator_llm = evaluator_llm.with_structured_output(output_schema)

def jd_evaluation(state:Jd)->Jd:
    query=f"Evaluate this job discription {state["tweet"]} for this topic {state["topic"]} and give a feedback "
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




#create the graph 
graph=StateGraph(Jd)
# add nodes 
graph.add_node("jd_genearation",jd_genearation)
graph.add_node("jd_evaluation",jd_evaluation)
graph.add_node("optimize_tweet",optimize_tweet)

#add edges 
graph.add_edge(START,"jd_genearation")
graph.add_edge("jd_genearation","jd_evaluation")

#add conditional edge
graph.add_conditional_edges("jd_evaluation", route_evaluation, {'approved': END, 'needs_improvement': 'optimize_tweet'})
graph.add_edge("optimize_tweet","jd_evaluation")


workflow = graph.compile()

workflow


#define initial state
initial_state = {
    "topic": "Data Science",
    "iteration": 1,
    "max_iteration": 5
}
result = workflow.invoke(initial_state)


print(result["iteration"])
