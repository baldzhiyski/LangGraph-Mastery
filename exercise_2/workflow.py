"""
workflow.py

LangGraph demo with:
- An LLM *supervisor* router that returns a STRICT, validated JSON route
  using Pydantic (no fragile JSON parsing).
- Two specialist LLM agents (Nutrition, Sleep) with function-calling tools.
- A manual tool loop for transparency, plus a small preflight validator to
  ensure we have enough arguments (fills defaults, checks required fields,
  and guards ranges) before actually invoking a tool.
- The run ends after the agent (no unconditional loop back to supervisor),
  preventing infinite recursion.

Prereqs:
  pip install -U langgraph langchain langchain-openai pydantic python-dotenv
  export OPENAI_API_KEY=...
"""

from typing import TypedDict, Literal, Dict, Any, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from dotenv import load_dotenv

# Load env vars (OPENAI_API_KEY, etc.)
load_dotenv()


# =========================
# 1) Shared Graph State
# =========================
class AppState(TypedDict, total=False):
    """
    State flowing through the graph. Extend as your app grows
    (e.g., user_profile, locale, metrics, etc.).
    """
    user_input: str
    route: Literal["nutrition", "sleep", "done"]
    result: str
    debug: Dict[str, Any]


# =========================
# 2) Tools (typed schemas)
# =========================
@tool
def tdee(
    weight_kg: float,
    height_cm: float,
    age: int,
    sex: Literal["male", "female"],
    activity: Literal["sedentary", "light", "moderate", "high", "athlete"]
) -> float:
    """
    Estimate daily energy expenditure (kcal) via Mifflin-St Jeor + activity multiplier.
    This is deterministic math; the LLM should call this when numeric inputs are provided.
    """
    mult = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "high": 1.725, "athlete": 1.9}[activity]
    bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + (5 if sex == "male" else -161)
    return round(bmr * mult, 1)


@tool
def macro_split(
    calories: float,
    weight_kg: float,
    protein_g_per_kg: float = 1.8
) -> Dict[str, float]:
    """
    Compute a simple macro plan:
    - Protein fixed by bodyweight (default 1.8 g/kg, adjustable).
    - Fat ~30% of calories.
    - Carbs = remainder.
    Returns grams for protein/fat/carbs.
    """
    protein_g = protein_g_per_kg * weight_kg
    protein_kcal = protein_g * 4
    fat_kcal = calories * 0.30
    fat_g = fat_kcal / 9
    carbs_kcal = max(calories - protein_kcal - fat_kcal, 0)
    carbs_g = carbs_kcal / 4
    return {"protein_g": round(protein_g, 1), "fat_g": round(fat_g, 1), "carbs_g": round(carbs_g, 1)}


@tool
def ideal_sleep_window(
    wake_time: str,
    chronotype: Literal["lark", "neutral", "owl"] = "neutral"
) -> Dict[str, str]:
    """
    Suggest bedtime ~8h before wake, adjusted by chronotype (+/- 30 min).
    wake_time must be 'HH:MM' (24h).
    """
    h, m = map(int, wake_time.split(":"))
    mins = h * 60 + m
    base = mins - (8 * 60)
    adjust = {"lark": -30, "neutral": 0, "owl": +30}[chronotype]
    bt = (base - adjust) % (24 * 60)
    h2, m2 = divmod(bt, 60)
    return {"bedtime": f"{h2:02d}:{m2:02d}", "target_sleep": "~8h"}


@tool
def sleep_debt(average_hours: float, target_hours: float = 8.0) -> float:
    """
    Weekly sleep debt (positive = shortfall). A quick quantification helper.
    """
    return round(max((target_hours - average_hours), 0.0) * 7, 2)


# =========================================
# 3) Tool arg preflight (robustness guard)
# =========================================
# Declarative argument rules: required fields, defaults for optional fields,
# and lightweight validators. This greatly improves reliability.
TOOL_ARG_RULES: Dict[str, Dict[str, Any]] = {
    "tdee": {
        "required": ["weight_kg", "height_cm", "age", "sex", "activity"],
        "defaults": {},
        "validators": {
            "weight_kg": lambda x: x > 0,
            "height_cm": lambda x: x > 0,
            "age":       lambda x: 0 < x < 120,
            "sex":       lambda x: x in ("male", "female"),
            "activity":  lambda x: x in ("sedentary", "light", "moderate", "high", "athlete"),
        },
    },
    "macro_split": {
        "required": ["calories", "weight_kg"],
        "defaults": {"protein_g_per_kg": 1.8},
        "validators": {
            "calories":          lambda x: x > 0,
            "weight_kg":         lambda x: x > 0,
            "protein_g_per_kg":  lambda x: 1.0 <= x <= 3.5,
        },
    },
    "ideal_sleep_window": {
        "required": ["wake_time"],
        "defaults": {"chronotype": "neutral"},
        "validators": {
            "wake_time": lambda s: isinstance(s, str) and ":" in s,
            "chronotype": lambda x: x in ("lark", "neutral", "owl"),
        },
    },
    "sleep_debt": {
        "required": ["average_hours"],
        "defaults": {"target_hours": 8.0},
        "validators": {
            "average_hours": lambda x: 0 <= x <= 14,
            "target_hours":  lambda x: 6 <= x <= 10,
        },
    },
}

def prepare_args(tool_name: str, args: dict) -> Dict[str, Any]:
    """
    Fill defaults for optional fields, check required fields,
    and validate simple ranges. Returns:
      {"ok": True, "args": <clean>}
      or
      {"ok": False, "error": "...", "args": <partial>}
    """
    spec = TOOL_ARG_RULES.get(tool_name, {})
    required = spec.get("required", [])
    defaults = spec.get("defaults", {})
    validators = spec.get("validators", {})

    args = dict(args or {})

    # Fill defaults for optional params
    for k, v in defaults.items():
        args.setdefault(k, v)

    # Check missing required
    missing = [k for k in required if k not in args]
    if missing:
        return {"ok": False, "error": f"Missing required fields: {', '.join(missing)}", "args": args}

    # Validate ranges/basic sanity
    invalid = []
    for k, validator in validators.items():
        if k in args:
            try:
                if not validator(args[k]):
                    invalid.append(k)
            except Exception:
                invalid.append(k)
    if invalid:
        return {"ok": False, "error": f"Invalid values for: {', '.join(invalid)}", "args": args}

    return {"ok": True, "args": args}


# ===========================================
# 4) Generic LLM Agent (with tool loop)
# ===========================================
def make_llm_agent(system_prompt: str, tools: list):
    """
    Creates a node:
      - Binds tools to LLM (function-calling).
      - Runs: LLM -> (tool_calls?) -> run tools -> LLM -> final text.
      - Uses `prepare_args` to ensure safe tool invocation.
      - Writes final text to state["result"].
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    def node(state: AppState) -> AppState:
        # Build conversation. You can prepend user profile/memory here if needed.
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["user_input"]),
        ]

        # First LLM response: could be final text or tool_calls.
        ai: AIMessage = llm.invoke(messages)
        messages.append(ai)

        # Allow up to 3 tool-calling rounds to remain predictable.
        for _ in range(3):
            if not getattr(ai, "tool_calls", None):
                break

            for call in ai.tool_calls:
                name = call["name"]
                raw_args = call.get("args") or {}

                # Preflight: fill defaults, ensure required args, validate ranges
                check = prepare_args(name, raw_args)
                if not check["ok"]:
                    # Feed a structured error back as the tool "result".
                    # The LLM will usually ask the user for the missing info next turn.
                    err_payload = {"status": "ERROR", "message": check["error"], "args_seen": raw_args}
                    messages.append(ToolMessage(tool_call_id=call["id"], name=name, content=str(err_payload)))
                    continue

                # Safe to invoke tool
                try:
                    result = tools_by_name[name].invoke(check["args"])
                except Exception as e:
                    result = {"status": "ERROR", "message": str(e)}

                # Return tool output to the LLM
                messages.append(ToolMessage(tool_call_id=call["id"], name=name, content=str(result)))

            # Let the LLM read tool results and either finish or request another call
            ai = llm.invoke(messages)
            messages.append(ai)

        # Final answer
        state["result"] = ai.content
        state.setdefault("debug", {})["agent_messages"] = len(messages)
        return state

    return node


# ============================
# 5) High-quality prompts
# ============================
NUTRITION_PROMPT = """
You are a precision Nutrition Coach.

Goals:
- Provide accurate, actionable guidance tailored to the user's inputs and goals.
- Use tools when numeric calculations are needed (tdee first, then macro_split if calories/weight are known).

Ask-vs-Assume Policy:
- REQUIRED for tdee(): weight_kg, height_cm, age, sex, activity (sedentary|light|moderate|high|athlete).
- If a REQUIRED field is missing, ask ONE concise question to collect it.
- If a NON-REQUIRED detail is missing, pick a safe default and state it.

Output:
- Be concise and structured. Show key numbers with units (kcal, g).
- State assumptions and next steps when relevant.
"""

SLEEP_PROMPT = """
You are a Sleep Coach.

Goals:
- Suggest a sustainable sleep schedule and optionally quantify sleep debt.
- Use tools: ideal_sleep_window (wake_time 'HH:MM', optional chronotype), sleep_debt (average_hours, optional target_hours).

Ask-vs-Assume Policy:
- If wake_time is missing, ask ONE short question.
- If chronotype unknown, assume 'neutral' and state it.
- For sleep_debt, if average_hours is unknown, either ask briefly or skip the computation.

Output:
- Be concise and supportive. Provide a clear bedtime target and 1–2 actionable tips.
"""


# =======================================
# 6) LLM Supervisor (Structured Routing)
# =======================================
class RouteDecision(BaseModel):
    """
    Strict schema for router output. The `with_structured_output` API ensures
    we either get a valid object or an error (no brittle string parsing).
    """
    route: Literal["nutrition", "sleep", "done"] = Field(
        description="Which agent to run next. Use 'done' if no agent is appropriate."
    )
    reason: Optional[str] = Field(default=None, description="Brief rationale for telemetry/debugging.")


ROUTER_PROMPT = """
You are the Supervisor Router.

You will be given:
- The user's latest message.
- The list of currently available agents: ["nutrition", "sleep"].

Task:
- Choose EXACTLY ONE route from: "nutrition", "sleep", or "done".
- Use "nutrition" for diet, calories, macros, protein, meal planning, supplements framed as nutrition, etc.
- Use "sleep" for bedtime, wake time, chronotype, insomnia, naps, sleep duration/debt, circadian issues, etc.
- Use "done" if none apply.

Return ONLY a JSON object matching the provided schema. Do not include advice or extra text.
"""

# LLM that returns a validated RouteDecision instance.
router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(RouteDecision)

def llm_supervisor(state: AppState) -> AppState:
    """
    Call the router LLM to decide the next node based on the user's message.
    Stores the route and an optional reason in state.debug for observability.
    """
    messages = [
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=f"User message: {state['user_input']}"),
    ]
    decision: RouteDecision = router_llm.invoke(messages)

    # Safety: constrain to allowed routes
    route = decision.route if decision.route in ("nutrition", "sleep", "done") else "done"
    state["route"] = route
    state.setdefault("debug", {})["router_reason"] = decision.reason
    return state

def decide_route(state: AppState) -> str:
    """LangGraph conditional edge resolver."""
    return state["route"]


# ============================
# 7) Build agents & graph
# ============================
nutrition_agent = make_llm_agent(NUTRITION_PROMPT, [tdee, macro_split])
sleep_agent = make_llm_agent(SLEEP_PROMPT, [ideal_sleep_window, sleep_debt])

graph = StateGraph(AppState)
graph.add_node("supervisor", llm_supervisor)   # LLM router
graph.add_node("nutrition", nutrition_agent)
graph.add_node("sleep", sleep_agent)

graph.set_entry_point("supervisor")

# Route from supervisor to exactly one of the destinations (or END)
graph.add_conditional_edges(
    "supervisor",
    decide_route,
    {"nutrition": "nutrition", "sleep": "sleep", "done": END},
)

# IMPORTANT: Do NOT loop back to supervisor by default.
# Each invoke handles exactly one hop: supervisor -> chosen agent -> END.
graph.add_edge("nutrition", END)
graph.add_edge("sleep", END)

# Compile the graph into a runnable app.
app = graph.compile()


# ============================
# 8) Demo runs
# ============================
if __name__ == "__main__":
    # Nutrition example (LLM should call tdee, then macro_split)
    out = app.invoke({
        "user_input": "Male, 82kg, 180cm, 29y, moderate activity. Plan 2200 kcal macros for fat loss."
    })
    print("\n=== [Nutrition] ===\n", out["result"])
    print("[Router reason]", out.get("debug", {}).get("router_reason"))

    # Sleep example (LLM should call ideal_sleep_window, may compute sleep_debt)
    out = app.invoke({
        "user_input": "I wake at 06:30 and average 6.5h. I'm an owl. What bedtime should I target?"
    })
    print("\n=== [Sleep] ===\n", out["result"])
    print("[Router reason]", out.get("debug", {}).get("router_reason"))

    # No matching domain -> done
    out = app.invoke({"user_input": "random topic"})
    print("\n=== [Done] ===\n", out.get("result", "Finished."))
    print("[Router reason]", out.get("debug", {}).get("router_reason"))
