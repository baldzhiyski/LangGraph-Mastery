from typing import TypedDict, List
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, END
from IPython.display import Image, display

class AgentState(TypedDict):
    messages: List[HumanMessage | AIMessage]

llm = OllamaLLM(model="llama3:8b")

def process_node(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response}")
    return {
        "messages": state["messages"] + [AIMessage(content=response)]
    }

def input_node(state: AgentState) -> AgentState:
    user_input = input("You: ")
    return {
        "messages": state["messages"] + [HumanMessage(content=user_input)]
    }

# ✅ Conditional logic after input
def check_exit(state: AgentState) -> str:
    last_msg = state["messages"][-1].content.strip().lower()
    return "exit" if last_msg == "exit" else "continue"

# Build the graph
graph = StateGraph(AgentState)
graph.add_node("input", input_node)
graph.add_node("process", process_node)

graph.set_entry_point("input")

# ✅ Conditional edge directly from input → END or process
graph.add_conditional_edges("input", check_exit, {
    "exit": END,
    "continue": "process"
})

# Simple edge process → input
graph.add_edge("process", "input")

# Compile and draw
agent = graph.compile()
png_bytes = agent.get_graph().draw_mermaid_png(output_file_path="exercise_1/final_graph.png")
display(Image(data=png_bytes))

# Start the loop
state: AgentState = {"messages": []}
agent.invoke(state)
