"""
agent_network.py

Fully-connected multi-agent network in LangGraph (stabilized + synthesis):
- Any agent can call any other agent next, or stop ("end").
- Each agent = (LLM + tools) + a small LLM "next-hop" router with structured output.
- Conditional edges form a complete graph.
- Stability guards: hop cap, per-agent visit caps, duplicate-output debounce, history-aware next-hop routing.
- NEW: final "synth" node merges all hop outputs into ONE coherent final answer.

Prereqs:
  pip install -U langgraph langchain langchain-openai pydantic python-dotenv
  export OPENAI_API_KEY=...
"""

from typing import TypedDict, Literal, Dict, Any, Optional, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
import hashlib

load_dotenv()

# --------------------------- Tuning knobs ---------------------------
DEFAULT_MAX_HOPS = 6                  # total agent hops allowed
MAX_VISITS_PER_AGENT = 2              # per-agent visit cap
DUP_HASH_PREFIX_LEN = 400             # chars to hash for duplicate detection
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# 1) Shared State
# -------------------------------------------------------------------
class NetState(TypedDict, total=False):
    user_input: str
    last_agent: Optional[str]
    next_agent: Optional[str]
    hops: int
    max_hops: int
    results: Dict[str, str]                 # {agent_name: last_text}
    trace: List[Dict[str, Any]]             # [{"agent": name, "text": "..."}...]
    debug: Dict[str, Any]
    visits: Dict[str, int]                  # {agent_name: count}
    last_text_hash_by_agent: Dict[str, str] # debounce per agent
    final_answer: str                       # <-- combined final answer 

# -------------------------------------------------------------------
# 2) Example Tools
# -------------------------------------------------------------------
@tool
def tdee(weight_kg: float, height_cm: float, age: int,
         sex: Literal["male", "female"], activity: Literal["sedentary", "light", "moderate", "high", "athlete"]) -> float:
    """Estimate kcal/day (Mifflin-St Jeor + activity multiplier)."""
    mult = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "high": 1.725, "athlete": 1.9}[activity]
    bmr = 10*weight_kg + 6.25*height_cm - 5*age + (5 if sex=="male" else -161)
    return round(bmr*mult, 1)

@tool
def ideal_sleep_window(wake_time: str, chronotype: Literal["lark","neutral","owl"]="neutral") -> Dict[str,str]:
    """Return target bedtime ~8h before wake, adjusted by chronotype +/-30 min."""
    h, m = map(int, wake_time.split(":"))
    mins = h*60+m
    base = mins - 8*60
    adj = {"lark": -30, "neutral": 0, "owl": +30}[chronotype]
    bt = (base - adj) % (24*60)
    hh, mm = divmod(bt, 60)
    return {"bedtime": f"{hh:02d}:{mm:02d}", "target_sleep":"~8h"}

@tool
def plan_training(split: Literal["full","upper-lower","push-pull-legs"]="full",
                  days_per_week: int = 3) -> List[str]:
    """Return a simple weekly training template."""
    if split == "full":
        return [f"Day {i+1}: Full-body strength + conditioning" for i in range(days_per_week)]
    if split == "upper-lower":
        return ["Upper strength", "Lower strength"] * (days_per_week//2)
    return ["Push", "Pull", "Legs"][:days_per_week]


# -------------------------------------------------------------------
# 3) Helpers
# -------------------------------------------------------------------
def short_hash(txt: str, n: int = DUP_HASH_PREFIX_LEN) -> str:
    """Short hash for duplicate-output detection."""
    return hashlib.sha1((txt[:n]).encode("utf-8")).hexdigest()[:12]


# -------------------------------------------------------------------
# 4) Agent kernel (LLM + tools + next-hop router)
# -------------------------------------------------------------------
class NextHop(BaseModel):
    next: Literal["nutrition", "sleep", "training", "end"] = Field(
        description="Which agent should run next (or 'end')."
    )
    reason: Optional[str] = Field(default=None, description="Short rationale for telemetry.")

def make_agent(
    name: str,
    system_prompt: str,
    tools: list,
    peer_names: List[str],
):
    """
    Creates a node function for one agent:
    - run LLM with tools (function-calling) and bounded tool loop,
    - write output + trace,
    - run hop router LLM (structured) to pick next agent,
    - apply guards: hop cap, visit caps, duplicate-output debounce, anti-oscillation.
    """
    work_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}
    hop_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(NextHop)

    HOP_PROMPT = f"""
You are the "{name}" agent deciding the NEXT HOP in a fully-connected agent network.
Peers you may choose next: {peer_names + ['end']}.

Context & Rules:
- Avoid oscillation between the same two agents.
- Respect the remaining hop budget; prefer "end" when budget is low.
- Prefer "end" if the user's goal seems satisfied or answers would repeat.
- Otherwise, pick exactly one peer that adds NEW, complementary value.
- Do NOT pick agents that were already visited too many times.

Return ONLY the JSON per schema. No extra text.
"""

    def node(state: NetState) -> NetState:
        # ----- global hop cap -----
        max_hops = state.get("max_hops", DEFAULT_MAX_HOPS)
        state["hops"] = state.get("hops", 0) + 1
        if state["hops"] > max_hops:
            state["next_agent"] = "end"
            state.setdefault("debug", {})["stop_reason"] = f"max_hops {max_hops} reached"
            return state

        # ----- per-agent visit cap -----
        visits = state.setdefault("visits", {})
        visits[name] = visits.get(name, 0) + 1
        if visits[name] > MAX_VISITS_PER_AGENT:
            state["last_agent"] = name
            state["next_agent"] = "end"
            state.setdefault("debug", {})[f"{name}_stop_reason"] = f"visit cap reached ({MAX_VISITS_PER_AGENT})"
            return state

        # ----- work phase (LLM + tools) -----
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["user_input"]),
        ]
        ai: AIMessage = work_llm.invoke(messages)
        messages.append(ai)

        for _ in range(3):  # bounded tool loop
            if not getattr(ai, "tool_calls", None):
                break
            for call in ai.tool_calls:
                tool_name = call["name"]
                args = call.get("args") or {}
                try:
                    result = tools_by_name[tool_name].invoke(args)
                except Exception as e:
                    result = {"status": "ERROR", "message": str(e)}
                messages.append(ToolMessage(tool_call_id=call["id"], name=tool_name, content=str(result)))
            ai = work_llm.invoke(messages)
            messages.append(ai)

        text = ai.content or ""
        state.setdefault("results", {})[name] = text
        state.setdefault("trace", []).append({"agent": name, "text": text})

        # ----- duplicate-output debounce (per agent) -----
        h = short_hash(text)
        last_hashes = state.setdefault("last_text_hash_by_agent", {})
        prev_h = last_hashes.get(name)
        last_hashes[name] = h
        if prev_h == h:
            state["last_agent"] = name
            state["next_agent"] = "end"
            state.setdefault("debug", {})[f"{name}_stop_reason"] = "duplicate output detected"
            return state

        # ----- decide next hop (structured) -----
        hops = state.get("hops", 0)
        hops_left = max(0, max_hops - hops)
        hop_msg = [
            SystemMessage(content=HOP_PROMPT),
            HumanMessage(content=(
                f"User message: {state['user_input']}\n"
                f"Your output summary (first {DUP_HASH_PREFIX_LEN} chars): {text[:DUP_HASH_PREFIX_LEN]}\n"
                f"Visits so far: {visits}\n"
                f"Hops used: {hops} / {max_hops} (left: {hops_left})\n"
                f"Last agent: {state.get('last_agent')}\n"
                f"Note: Prefer 'end' if goals are met, answers would repeat, or budget is low.\n"
            )),
        ]
        decision: NextHop = hop_llm.invoke(hop_msg)

        allowed = set(peer_names) | {"end"}
        nxt = decision.next if decision.next in allowed else "end"

        # anti-oscillation & capacity
        if nxt == state.get("last_agent"):
            nxt = "end"
        if nxt in visits and visits.get(nxt, 0) >= MAX_VISITS_PER_AGENT:
            nxt = "end"

        state["last_agent"] = name
        state["next_agent"] = nxt
        state.setdefault("debug", {})[f"{name}_hop_reason"] = decision.reason
        return state

    return node


# -------------------------------------------------------------------
# 5) Synthesizer node (combine all hop outputs into one answer)
# -------------------------------------------------------------------
def synth_node(state: NetState) -> NetState:
    """
    Merge all agent outputs into a single coherent final answer.

    You can:
      - do a simple deterministic join, or
      - use an LLM to deduplicate/organize/format nicely (we do that here).
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Build a compact transcript of the agent hops
    parts = []
    for step in state.get("trace", []):
        agent = step.get("agent", "agent")
        text = (step.get("text") or "").strip()
        if text:
            parts.append(f"[{agent}] {text}")

    # Guard: if nothing to merge, just echo last agent’s output
    merged = "\n\n".join(parts) if parts else state.get("results", {}).get(state.get("last_agent", ""), "")

    sys = SystemMessage(content=(
        "You are a coordinator. Produce ONE clear, concise final answer that integrates the agent outputs. "
        "Remove duplicates, resolve conflicts, and present an actionable plan. Use brief bullets and headings."
    ))
    hum = HumanMessage(content=(
        f"User message:\n{state['user_input']}\n\n"
        f"Agent hops (ordered):\n{merged}\n\n"
        "Write the final answer now."
    ))
    out: AIMessage = llm.invoke([sys, hum])
    state["final_answer"] = out.content.strip() if out.content else merged
    # Stop after synthesis
    return state


# -------------------------------------------------------------------
# 6) Agent catalog
# -------------------------------------------------------------------
AGENTS_DEF = {
    "nutrition": {
        "prompt": (
            "You are the Nutrition agent. Provide precise, actionable diet advice.\n"
            "- If numbers provided, compute TDEE. If user has kcal target, propose macros.\n"
            "- Be concise. Use tools where helpful."
        ),
        "tools": [tdee],
    },
    "sleep": {
        "prompt": (
            "You are the Sleep agent. Optimize sleep timing and hygiene.\n"
            "- If wake time provided, propose an ideal bedtime; offer 1–2 tips."
        ),
        "tools": [ideal_sleep_window],
    },
    "training": {
        "prompt": (
            "You are the Training agent. Propose sensible programming templates.\n"
            "- Offer a weekly structure based on split and days per week."
        ),
        "tools": [plan_training],
    },
}


# -------------------------------------------------------------------
# 7) Build graph (fully-connected + synth)
# -------------------------------------------------------------------
graph = StateGraph(NetState)

# Add agent nodes
for agent_name, cfg in AGENTS_DEF.items():
    peers = [n for n in AGENTS_DEF.keys() if n != agent_name]
    graph.add_node(agent_name, make_agent(agent_name, cfg["prompt"], cfg["tools"], peers))

# Add synthesis node
graph.add_node("synth", synth_node)

# Entry point (could replace with an entry router if you like)
START_AGENT = "nutrition"
graph.set_entry_point(START_AGENT)

def decide_next(state: NetState) -> str:
    """Resolver used by conditional edges to pick the next node key."""
    return state.get("next_agent", "end")

# Wire: from each agent, go to any agent or to "synth" (instead of END)
for src in AGENTS_DEF.keys():
    destinations = {dst: dst for dst in AGENTS_DEF.keys()}
    destinations["end"] = "synth"   # <-- send 'end' decisions to synth node
    graph.add_conditional_edges(src, decide_next, destinations)

# After synth, terminate
graph.add_edge("synth", END)

# Compile
app = graph.compile()


# -------------------------------------------------------------------
# 8) Demo
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Demo: fully-connected agent network (stabilized + synthesis) ===")

    # Example 1
    out = app.invoke({
        "user_input": "Male, 34y, 82kg, 180cm, moderate activity. I’m an owl and usually wake at 06:30 for work. Target fat loss ~0.5 kg/week and better morning energy. Suggest an ideal bedtime, plus a gentle 2-day training routine I can do at home (kettlebells + pull-up bar). I’m lactose-intolerant and prefer ~2200 kcal/day. Please keep it practical.",
        "max_hops": 5,
    })
    print("\n[TRACE]")
    for step in out.get("trace", []):
        print(f"- {step['agent']}: {step['text'][:120]}...")
    print("\n[FINAL ANSWER]\n", out.get("final_answer", "<no final>"))
    print("\n[FINAL STATE]")
    print("last_agent:", out.get("last_agent"))
    print("next_agent:", out.get("next_agent"))
    print("hops:", out.get("hops"))
    print("stop_reason:", out.get("debug", {}).get("stop_reason"))
    print("visit_caps:", out.get("visits"))