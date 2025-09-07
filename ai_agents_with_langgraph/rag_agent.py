from dotenv import load_dotenv
import os

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_chroma import Chroma
from langchain_core.tools import tool

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

embendings = OpenAIEmbeddings(model="text-embedding-3-small")

pdf_path = "Bernard_Resume.pdf";


if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise

# chunking
text_slitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

pages_split = text_slitter.split_documents(pages)

persist_directory = "./ChromaDB"
collection_name = 'resume'

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    vector_store = Chroma.from_documents(
        documents=pages_split,
        embedding=embendings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Created ChromaDB vector store")
except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

@tool
def retriever_tool(query: str) -> str:
    """This tool searches and returns information from the resume PDF document"""
    docs = retriever.invoke(query)

    if not docs:
        return "I found not relevant information in the resume PDF document."
    
    results = []

    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}: \n{doc.page_content}")

    return "\n\n".join(results)


tools = [retriever_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState) -> AgentState:
    """Check if the last message contains tool calls."""
    result = state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0

system_prompt = """
You are an intelligent AI assistant who answers questions about a resume based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the person's bio, education and experience. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

tools_dict = { our_tool.name: our_tool for our_tool in tools}

def call_llm(state: AgentState) -> AgentState:
    """Function to all the LLM with the current state"""
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {"messages": [message]}

def take_action(state: AgentState):
    """Execute tool calls from the LLM's response"""

    tool_calls = state["messages"][-1].tool_calls

    results = []

    for tool in tool_calls:
        print(f"Calling tool: {tool['name']} with query {tool['args'].get('query', 'No query provided')}")

        if not tool["name"] in tools_dict:
            print(f"\nTool: {tool['name']} does not exist.")
            result = "Incorrect tool name. Please retry and select tool from the list of available tools."
        else:
            result = tools_dict[tool["name"]].invoke(tool["args"].get("query", ""))
            print(f"Result length: {len(str(result))}")\
            
        results.append(ToolMessage(tool_call_id=tool["id"], name=tool["name"], content=str(result)))

    print("Tools execution compleet. Back to the model!")
    return {"messages": results}

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)
graph.add_edge("retriever_agent", "llm")
graph.add_conditional_edges(
    "llm",
    should_continue,
    { True: "retriever_agent", False: END}
)
graph.set_entry_point("llm")

agent = graph.compile()

def run_agent():
    print("\n=== RESUME AGENT ===")

    while True:
        user_input = input("\nInput: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        messages = [HumanMessage(content=user_input)]

        result = agent.invoke({"messages": messages})

        print("\n=== ANSWER ===")
        print(result["messages"][-1].content)


run_agent()