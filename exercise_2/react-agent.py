from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_ollama import OllamaLLM
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from IPython.display import Image, display

load_dotenv()

# 1. Define State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# 2. LLM + Tools
model = OllamaLLM(model="llama3:8b")

@tool
def add(a: int, b: int):
    """Add two numbers"""
    return a + b

@tool
def subtract(a: int, b: int):
    """Subtract two numbers"""
    return a - b

@tool
def multiply(a: int, b: int):
    """Multiply two numbers"""
    return a * b

tools = [add, subtract, multiply]

# 3. Node that invokes LLM
def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are my AI assistant. Use tools if needed.")
    messages = [system_prompt] + state["messages"]
    response = model.invoke(messages)
    return {"messages": state["messages"] + [response]}

# 4. Conditional logic
def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "continue"
    return "end"

# 5. Graph definition
graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)
graph.add_node("tools", ToolNode(tools=tools))
graph.set_entry_point("our_agent")

graph.add_conditional_edges("our_agent", should_continue, {
    "continue": "tools",
    "end": END,
})
graph.add_edge("tools", "our_agent")

app = graph.compile()

# 6. Draw Graph
png_bytes = app.get_graph().draw_mermaid_png(output_file_path="final_graph.png")
display(Image(data=png_bytes))

# 7. Run Stream
def print_stream(stream):
    for step in stream:
        last_msg = step["messages"][-1]
        if hasattr(last_msg, "pretty_print"):
            last_msg.pretty_print()
        else:
            print(last_msg)

inputs = {"messages": [HumanMessage(content="Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")]}
print_stream(app.stream(inputs, stream_mode="values"))
