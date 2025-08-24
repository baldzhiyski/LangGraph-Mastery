from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage # Base class for all message types (human, AI, system, tool, etc.)
from langchain_core.messages import ToolMessage # Special message returned when a tool finishes execution
from langchain_core.messages import SystemMessage # Message for providing system-level instructions to the LLM
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


load_dotenv()  # Load your environment variables (e.g., OPENAI_API_KEY)

# ---------------------------
# 1. Define the STATE
# ---------------------------
class AgentState(TypedDict):
    # The state holds a list of messages that accumulate during the run.
    # Annotated[...] + add_messages ensures that new messages are automatically appended.
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ---------------------------
# 2. Define TOOLS
# ---------------------------
@tool
def add(a: int, b:int):
    """This is an addition function that adds 2 numbers together"""
    return a + b 

@tool
def subtract(a: int, b: int):
    """Subtraction function"""
    return a - b

@tool
def multiply(a: int, b: int):
    """Multiplication function"""
    return a * b

# Put all tools in a list
tools = [add, subtract, multiply]


# ---------------------------
# 3. LLM SETUP
# ---------------------------
# Create an OpenAI chat model and tell it which tools it may call
model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)


# ---------------------------
# 4. Agent NODE (LLM step)
# ---------------------------
def model_call(state:AgentState) -> AgentState:
    # Add a system instruction to guide the LLM
    system_prompt = SystemMessage(content=
        "You are my AI assistant, please answer my query to the best of your ability."
    )
    # Call the LLM with [system_prompt] + current conversation
    response = model.invoke([system_prompt] + state["messages"])
    # Return a new state: the new AIMessage is appended to the list of messages
    return {"messages": [response]}


# ---------------------------
# 5. CONTROL LOGIC
# ---------------------------
def should_continue(state: AgentState): 
    # Look at the last message
    messages = state["messages"]
    last_message = messages[-1]
    # If the AI requested a tool (AIMessage.tool_calls not empty) → continue to tools
    if last_message.tool_calls: 
        return "continue"
    # Otherwise, the AI produced a final answer → end the graph
    return "end"
    

# ---------------------------
# 6. GRAPH DEFINITION
# ---------------------------
graph = StateGraph(AgentState)  # Create a graph whose state type is AgentState

# Add nodes to the graph
graph.add_node("our_agent", model_call)  # Node where the LLM runs

tool_node = ToolNode(tools=tools)        # A prebuilt node that runs tools when requested
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")       # Start execution in the LLM node

# Conditional edges: after the agent runs, decide whether to END or go to tools
graph.add_conditional_edges(
    "our_agent",
    should_continue,    # function that decides where to go next
    {
        "continue": "tools",  # go to tools node
        "end": END,           # stop the graph
    },
)

# After tools finish, return to the agent so it can process tool outputs
graph.add_edge("tools", "our_agent")

# Compile the graph into an executable app
app = graph.compile()


# ---------------------------
# 7. HELPER: Pretty print stream
# ---------------------------
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]  # Look at the latest message
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()   # Nicely formatted print of AI/Human/Tool messages


# ---------------------------
# 8. RUN EXAMPLE
# ---------------------------
inputs = {
    "messages": [
        ("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")
    ]
}

# Stream the execution step by step
print_stream(app.stream(inputs, stream_mode="values"))
