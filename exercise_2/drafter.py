# drafter.py
from typing import Annotated, Sequence, TypedDict, Literal
import json
from dotenv import load_dotenv

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


load_dotenv()  # expects OPENAI_API_KEY in your environment

# =========================
# 1) STATE (flows through the graph)
# =========================
class AgentState(TypedDict):
    # Conversation history (Human/AI/Tool/System messages).
    # 'add_messages' tells LangGraph to APPEND new messages returned by nodes.
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # The authoritative document content (no globals).
    document: str
    # Signal to end the run (set True after save or if we choose to stop).
    finished: bool


# =========================
# 2) TOOLS (return JSON instructions; do NOT mutate state directly)
# =========================
@tool
def update(content: str) -> str:
    """Replace the entire document content with 'content'.
    Returns a JSON instruction that another node will apply."""
    return json.dumps({
        "ok": True,
        "event": "update",
        "content": content,
        "message": "Document updated."
    })

@tool
def append(text: str) -> str:
    """Append 'text' to the document.
    Returns a JSON instruction that another node will apply."""
    return json.dumps({
        "ok": True,
        "event": "append",
        "text": text,
        "message": "Text appended."
    })

@tool
def show() -> str:
    """Request to show the current document to the user (no state change)."""
    return json.dumps({
        "ok": True,
        "event": "show",
        "message": "Show current document."
    })

@tool
def save(filename: str) -> str:
    """Request to save the current document to 'filename' (applied later)."""
    if not filename.endswith(".txt"):
        filename = filename + ".txt"
    return json.dumps({
        "ok": True,
        "event": "save",
        "filename": filename,
        "message": "Please save the current doc to this path."
    })

tools = [update, append, show, save]


# =========================
# 3) MODEL (bound to tools)
# =========================
# Tip: you can add tool_choice="required" if you want to force tool calls each turn.
model = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)


# =========================
# 4) AGENT NODE (LLM decides which tools to call)
# =========================
def agent_node(state: AgentState) -> AgentState:
    """
    - Adds a SystemMessage with the current doc snapshot and rules.
    - Invokes the LLM with the existing history.
    - If the LLM calls tools, ToolNode will execute them next.
    """
    system_prompt = SystemMessage(content=(
        "You are Drafter, a helpful writing assistant.\n"
        "Rules:\n"
        "1) Only call 'update' when the final document content actually CHANGES.\n"
        "2) Use 'append' for small additions to the end.\n"
        "3) After a successful update/append, call 'show' ONCE to reveal the document, then stop.\n"
        "4) When the user asks to persist, call 'save(filename=...)'.\n"
        "5) Do NOT output the document yourself; use 'show' to display it.\n"
        "6) Avoid repeating the same 'update' with identical content.\n\n"
        f"Current document snapshot:\n{state.get('document', '') or '<empty>'}\n"
    ))

    ai = model.invoke([system_prompt] + list(state["messages"]))
    return {"messages": [ai]}


# =========================
# 5) APPLY TOOL RESULTS (mutates state based on ToolMessages)
# =========================
def apply_tool_results(state: AgentState) -> AgentState:
    """
    Parse ToolMessages emitted since the last AI tool-call and:
      - apply 'update' only if content changed (prevents redundant loops),
      - apply 'append' by concatenating text,
      - perform 'save' (write to disk) and set finished=True,
      - 'show' doesn't change state (router will end after a single show).
    """
    msgs = list(state["messages"])

    # Find the last AIMessage that actually issued tool calls
    last_ai_idx = None
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            last_ai_idx = i
            break

    if last_ai_idx is None:
        return {}  # nothing to apply

    new_doc = state["document"]
    end_now = state["finished"]

    i = last_ai_idx + 1
    while i < len(msgs):
        tm = msgs[i]
        i += 1
        if not isinstance(tm, ToolMessage):
            continue

        # ToolNode always wraps tool output as text -> parse JSON if possible
        try:
            payload = json.loads(tm.content)
        except Exception:
            continue

        event = payload.get("event")

        if event == "update":
            incoming = payload.get("content", new_doc)
            # ✅ Only apply if the content is actually different
            if incoming != new_doc:
                new_doc = incoming

        elif event == "append":
            new_doc = (new_doc or "") + payload.get("text", "")

        elif event == "save":
            filename = payload.get("filename", "document.txt")
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(new_doc or "")
                # Mark finished so router ends the flow
                end_now = True
            except Exception as e:
                # You could append an error message here if desired
                pass

        elif event == "show":
            # No state change; router will end after a single 'show' (see should_continue)
            pass

    return {"document": new_doc, "finished": end_now}


# =========================
# 6) ROUTER (decide whether to continue or end)
# =========================
def should_continue(state: AgentState) -> Literal["continue", "end"]:
    # End immediately if we have saved successfully
    if state.get("finished"):
        return "end"

    # If the most recent message is a 'show' ToolMessage, end to avoid show→show loops.
    last = state["messages"][-1] if state["messages"] else None
    if isinstance(last, ToolMessage):
        try:
            payload = json.loads(last.content)
            if payload.get("event") == "show":
                return "end"
        except Exception:
            pass

    return "continue"


# =========================
# 7) GRAPH WIRING
# =========================
graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))  # executes tool calls found in the last AIMessage
graph.add_node("apply", apply_tool_results)

graph.set_entry_point("agent")
graph.add_edge("agent", "tools")
graph.add_edge("tools", "apply")
graph.add_conditional_edges("apply", should_continue, {"continue": "agent", "end": END})

app = graph.compile()


# =========================
# 8) DEMO RUNNER (no input(); provide Human messages programmatically)
# =========================
def print_tool_results(messages: Sequence[BaseMessage]) -> None:
    for m in messages:
        if isinstance(m, ToolMessage):
            print("🛠️ TOOL RESULT:", m.content)

def demo_run():
    print("\n===== DRAFTER (DEMO) =====")
    state: AgentState = {"messages": [], "document": "", "finished": False}

    # 1) User asks to update and show
    state["messages"] += [HumanMessage(content="Update the document with: 'Hello world!' and then show.")]
    for step in app.stream(state, stream_mode="values"):
        state = step  # keep newest state
        print_tool_results(state["messages"])
        if state.get("finished"):
            break

    print("\nDocument now:", repr(state["document"]))

    # 2) User appends and shows it
    state["messages"] += [HumanMessage(content="Append: '\\nThis is a second line.' then show.")]
    for step in app.stream(state, stream_mode="values"):
        state = step
        print_tool_results(state["messages"])
        if state.get("finished"):
            break

    print("\nDocument now:", repr(state["document"]))

    # 3) User asks to save
    state["messages"] += [HumanMessage(content="Save as hello_world.txt")]
    for step in app.stream(state, stream_mode="values"):
        state = step
        print_tool_results(state["messages"])
        if state.get("finished"):
            break

    print("\nDocument finally:", repr(state["document"]))
    print("Finished:", state["finished"])
    print("===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    demo_run()
