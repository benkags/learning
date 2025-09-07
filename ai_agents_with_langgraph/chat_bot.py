from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatOpenAI(model="gpt-4o")

def process(state: AgentState) -> AgentState:
    """This node will respond to the requests made by the user"""
    response = llm.invoke(state["messages"])

    state["messages"].append(HumanMessage(content=response.content))
    print(f"\nAI: {response.content}")

    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

conversation_history = [] # List[Union[HumanMessage, AIMessage]]

user_input = input("Enter: ")

while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("Enter: ")

