"""
Hierarchical multi-agent demo (Training · Nutrition · Biohacking)
with tools, true human-in-the-loop via LangGraph interrupts,
multi-hop routing (agent → supervisor → agent), tracing, and lightweight memory.

This version uses Command(resume=...) to continue after interrupts and
handles BOTH cases:
  1) The model tries to call a tool but is missing required arguments.
     → We feed a structured tool error to the model, let it write a clarifying question,
       then PAUSE the graph with `interrupt(...)`. When you resume with an answer,
       execution continues inside the SAME agent and tools can be re-tried.
  2) The model ends a turn with a clarifying question WITHOUT attempting a tool call.
     → We detect that (heuristic `_looks_like_question`) and PAUSE the graph the same way.

The file demonstrates:
  - A strict tool-argument preflight (no silent defaults). This forces the LLM
    to ask for missing information in a predictable loop.
  - A *supervisor router* that chooses exactly one domain agent for each "hop".
  - A *post-run router* that decides whether to hop to another agent or end.
  - A *human review* gate implemented as a dynamic interrupt.
  - Persistent execution with a checkpointer (MemorySaver). You can switch to
    SqliteSaver/Redis for durability across process restarts.
  - A CLI controller that shows how to catch interrupts and resume with Command(resume=...).

Prereqs:
  pip install -U langgraph langchain langchain-openai pydantic python-dotenv
  export OPENAI_API_KEY=...
"""

from typing import TypedDict, Literal, Dict, Any, Optional, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangGraph core: graph building, terminal node, and interrupts
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command  # `interrupt` pauses; `Command(resume=...)` resumes

# LLM + tool primitives (LangChain)
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

load_dotenv()


# ======================================================================================
# 1) SHARED GRAPH STATE
# --------------------------------------------------------------------------------------
# The LangGraph "state" is a dict-like object threaded through every node. We define a
# TypedDict so it's explicit and type-checked. Add fields as your app grows: user profile,
# locale, A/B flags, etc.
# ======================================================================================

class AppState(TypedDict, total=False):
    # The user's latest natural-language input (what each agent reads)
    user_input: str

    # The next route (agent) that should run; written by supervisor/post_router
    route: Literal["training", "nutrition", "biohacking", "done"]

    # The latest agent's final text (what the human ultimately sees/approves)
    result: str

    # Debug/telemetry (router rationales, error strings, message counts, etc.)
    debug: Dict[str, Any]

    # ---- Lightweight memory / tracing (purely illustrative) ----
    # Which agents we've visited in order (e.g., ["nutrition", "training", ...])
    route_history: List[str]
    # Count of visits per agent (e.g., {"nutrition": 2, "training": 1})
    agent_visit_counts: Dict[str, int]
    # Append a dict per agent run: tools called, args/results, short output preview
    traces: List[Dict[str, Any]]
    # Number of agent "hops" so far in this run
    hops: int
    # Whether post_router may route to a second agent
    allow_multi: bool
    # Safety stop so we don't loop forever
    max_hops: int
    # Prevent excessive repeat of the same agent
    repeat_limit: int


def _init_memory(state: AppState) -> None:
    """
    Idempotently initialize memory fields. Called in nodes that rely on these keys.
    """
    state.setdefault("route_history", [])
    state.setdefault("agent_visit_counts", {})
    state.setdefault("traces", [])
    state.setdefault("hops", 0)
    state.setdefault("allow_multi", True)
    state.setdefault("max_hops", 3)       # total agent hops before review
    state.setdefault("repeat_limit", 2)   # max times to visit the same agent


def _interrupt_payload(state):
    """
    Unwrap the interrupt payload across shapes:
      - state["__interrupt__"] may be an Interrupt object, a dict, or a list of either.
    We return the underlying `value` dict passed to `interrupt(...)`.
    """
    intr = state.get("__interrupt__")
    if not intr:
        return None
    if isinstance(intr, list) and intr:
        intr = intr[0]
    # Object form: Interrupt(value=..., id=...)
    if hasattr(intr, "value"):
        return intr.value
    # Dict form: {"value": {...}}
    if isinstance(intr, dict) and "value" in intr:
        return intr["value"]
    # Fallback (shouldn't happen in normal cases)
    return intr


# ======================================================================================
# 2) TOOLS
# --------------------------------------------------------------------------------------
# Tools are deterministic, idempotent functions for math/planning, bound to the LLM via
# tool calling. The LLM must supply the required arguments; we DO NOT guess defaults.
# That means the model will need to ask the user for missing inputs (which we enforce).
# ======================================================================================

@tool
def tdee(
    weight_kg: float,
    height_cm: float,
    age: int,
    sex: Literal["male", "female"],
    activity: Literal["sedentary", "light", "moderate", "high", "athlete"],
) -> float:
    """
    Estimate Total Daily Energy Expenditure (kcal) via Mifflin-St Jeor BMR + activity multiplier.

    This is deterministic math; no model judgement involved.
    """
    mult = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "high": 1.725, "athlete": 1.9}[activity]
    bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + (5 if sex == "male" else -161)
    return round(bmr * mult, 1)


@tool
def macro_plan(calories: float, weight_kg: float, protein_g_per_kg: float) -> Dict[str, float]:
    """
    Simple macro plan (STRICT: no defaults).
      - Protein = protein_g_per_kg * bodyweight
      - Fat     = 30% of calories
      - Carbs   = remainder

    Returns grams for protein/fat/carbs, rounded to one decimal.
    """
    protein_g = protein_g_per_kg * weight_kg
    fat_kcal = 0.30 * calories
    carbs_kcal = max(calories - protein_g * 4 - fat_kcal, 0)
    return {"protein_g": round(protein_g, 1), "fat_g": round(fat_kcal / 9, 1), "carbs_g": round(carbs_kcal / 4, 1)}


@tool
def build_workout(
    days_per_week: int,
    experience: Literal["beginner", "intermediate", "advanced"],
    goal: Literal["fat_loss", "muscle_gain", "performance"],
) -> Dict[str, Any]:
    """
    Mocked weekly template generator. In a real system you'd return a richer plan.

    Examples:
      - 3 days → ["Full Body A", "Full Body B", "Full Body C"]
      - 4 days → ["Upper", "Lower", "Push", "Pull"]
      - 5 days → ["Upper", "Lower", "Push", "Pull", "Full Body"]
    """
    split = {
        3: ["Full Body A", "Full Body B", "Full Body C"],
        4: ["Upper", "Lower", "Push", "Pull"],
        5: ["Upper", "Lower", "Push", "Pull", "Full Body"],
    }.get(days_per_week, ["Full Body"] * days_per_week)
    note = f"{experience.title()} plan focused on {goal.replace('_', ' ')}."
    return {"days": days_per_week, "template": split, "note": note}


@tool
def readiness_flag(hrv_ms: float, resting_hr: float) -> str:
    """
    A toy readiness classifier from HRV + RHR. Real use = more nuanced.
      - ready   : HRV ≥ 60 and RHR ≤ 60
      - normal  : HRV ≥ 45 and RHR ≤ 70
      - caution : otherwise
    """
    if hrv_ms >= 60 and resting_hr <= 60:
        return "ready"
    if hrv_ms >= 45 and resting_hr <= 70:
        return "normal"
    return "caution"


# --------------------------------------------------------------------------------------
# 2b) STRICT TOOL-ARG PRECHECK ("preflight")
# --------------------------------------------------------------------------------------
# We force the LLM to ask the user when required fields are missing. No silent defaults.
# Why? It prevents the model from making up numbers and produces predictable, correct
# tool invocations.
# --------------------------------------------------------------------------------------

TOOL_RULES: Dict[str, Dict[str, Any]] = {
    "tdee": {"required": ["weight_kg", "height_cm", "age", "sex", "activity"]},
    "macro_plan": {"required": ["calories", "weight_kg", "protein_g_per_kg"]},
    "build_workout": {"required": ["days_per_week", "experience", "goal"]},
    "readiness_flag": {"required": ["hrv_ms", "resting_hr"]},
}


def prepare_args(tool_name: str, args: dict) -> Dict[str, Any]:
    """
    Validate tool arguments against TOOL_RULES.

    Returns:
      { "ok": True, "args": <clean args> }
      or
      { "ok": False, "missing": [...], "error": "Missing: ...", "args": <partial> }
    """
    spec = TOOL_RULES.get(tool_name, {})
    args = dict(args or {})
    missing = [k for k in spec.get("required", []) if k not in args]
    if missing:
        return {"ok": False, "missing": missing, "args": args, "error": f"Missing: {', '.join(missing)}"}
    return {"ok": True, "args": args}


# ======================================================================================
# 3) GENERIC LLM AGENT
# --------------------------------------------------------------------------------------
# Each agent:
#   - Receives the user's last input + its role prompt.
#   - May issue *tool calls* with JSON args.
#   - If required tool args are missing, we:
#       (a) place a ToolMessage error → model writes a clarifying question
#       (b) raise a LangGraph *interrupt* with that question
#       (c) on resume, inject the human's answer, then let the model re-plan
#   - If the model asks a clarifying question without calling a tool at all,
#     we interrupt on that too (heuristic `_looks_like_question`), then resume.
#
# Notes:
#   - `max_rounds` bounds (LLM ↔ tools) inner loop so you can't get infinite loops.
#   - We keep a `traces` list to print what each agent did (tools used, first call, preview).
# ======================================================================================

def _looks_like_question(text: str) -> bool:
    """
    Heuristic to detect a clarifying question when the model didn't call a tool.
    Tuning this can change how often you interrupt vs. continue.
    """
    if not text:
        return False
    t = text.strip().lower()
    if t.endswith("?") or "?" in t:
        return True
    cues = [
        "what is your",
        "what's your",
        "please provide",
        "could you",
        "do you",
        "which",
        "how many",
        "how much",
        "when do you",
        "i need to know",
        "i need",
        "missing",
        "provide",
    ]
    return any(c in t for c in cues)


def make_agent(agent_name: str, system_prompt: str, tools: list):
    """
    Factory that returns a LangGraph node function for a role-specific agent.

    The returned node:
      - Binds `tools` to the chat model (function calling).
      - Runs up to `max_rounds` of plan→(tool)→reflect cycles.
      - Interrupts on missing inputs (preflight fail) OR pure clarifying questions.
      - Appends a trace with tools used and short output preview.
    """
    # Bind tool schema to the model so it can call them with structured arguments.
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    def node(state: AppState) -> AppState:
        _init_memory(state)

        # Seed the dialogue for this agent: role system prompt + user's last text.
        msgs = [SystemMessage(content=system_prompt), HumanMessage(content=state["user_input"])]

        # First model turn: it may directly talk, or may propose tool_calls.
        ai: AIMessage = llm.invoke(msgs)
        msgs.append(ai)

        tools_log: List[Dict[str, Any]] = []
        max_rounds = 8  # raise if your tasks are more complex
        any_success = False  # set True once ANY tool call succeeds

        rounds = 0
        while rounds < max_rounds:
            rounds += 1

            # ---- CASE A: the model wants to call one or more tools ----
            if getattr(ai, "tool_calls", None):
                for call in ai.tool_calls:
                    name, raw_args = call["name"], (call.get("args") or {})
                    check = prepare_args(name, raw_args)

                    # Missing args → teach the model to ask, then interrupt to get a human answer
                    if not check["ok"]:
                        # (1) Feed a structured error back to the model as a ToolMessage.
                        #     This gives the model explicit context about what's missing.
                        error_payload = {
                            "status": "ERROR",
                            "message": check["error"],
                            "missing": check["missing"],
                            "args_seen": raw_args,
                        }
                        msgs.append(ToolMessage(tool_call_id=call["id"], name=name, content=str(error_payload)))

                        # (2) Ask the model to respond to that error (it should formulate a short question).
                        ai = llm.invoke(msgs)
                        msgs.append(ai)
                        question_text = (ai.content or "").strip() or f"Please provide for {name}: {', '.join(check['missing'])}."

                        # (3) Pause here and wait for a human answer.
                        answer = interrupt(
                            {
                                "type": "missing_args",
                                "agent": agent_name,
                                "tool": name,
                                "question": question_text,
                                "missing": check["missing"],
                            }
                        )

                        # (4) Inject the human answer and let the model re-plan (it may now re-call tools).
                        msgs.append(HumanMessage(content=str(answer)))
                        ai = llm.invoke(msgs)
                        msgs.append(ai)

                        tools_log.append(
                            {
                                "name": name,
                                "args": raw_args,
                                "error": check["error"],
                                "interrupt_question": question_text,
                                "interrupt_answer": answer,
                            }
                        )
                        # Continue loop so the model can now issue valid tool calls with completed info.
                        continue

                    # (Happy path) Safe to invoke tool.
                    try:
                        out = tools_by_name[name].invoke(check["args"])
                        if not (isinstance(out, dict) and out.get("status") == "ERROR"):
                            any_success = True
                        tools_log.append({"name": name, "args": check["args"], "result": out})
                    except Exception as e:
                        out = {"status": "ERROR", "message": str(e)}
                        tools_log.append({"name": name, "args": check["args"], "error": str(e)})

                    # Return tool output to the model so it can integrate the result into the final answer or next calls.
                    msgs.append(ToolMessage(tool_call_id=call["id"], name=name, content=str(out)))

                # After completing all tool calls for this turn, let the model reflect/respond and loop again.
                ai = llm.invoke(msgs)
                msgs.append(ai)
                continue

            # ---- CASE B: no tool calls issued this turn ----
            # If we're *still* missing key info (no successful tool yet) and the model asked a question,
            # treat it as a clarification request and interrupt here to capture the user's answer.
            if _looks_like_question(ai.content or "") and not any_success:
                question_text = (ai.content or "").strip()
                answer = interrupt(
                    {
                        "type": "missing_args",
                        "agent": agent_name,
                        "tool": "unknown",  # clarification not tied to a specific tool
                        "question": question_text,
                        "missing": [],
                    }
                )
                msgs.append(HumanMessage(content=str(answer)))
                ai = llm.invoke(msgs)
                msgs.append(ai)
                # Loop so the model can proceed to tools using the new info.
                continue

            # Otherwise, nothing else to do in this agent; finalize.
            break

        # ---- Finalize this agent hop: write outputs and memory ----
        state["result"] = ai.content
        state["hops"] += 1
        state["route_history"].append(agent_name)
        state["agent_visit_counts"][agent_name] = state["agent_visit_counts"].get(agent_name, 0) + 1
        state["traces"].append(
            {
                "agent": agent_name,
                "input": state["user_input"],
                "tools_used": tools_log,
                "output_preview": (ai.content or "")[:600],
            }
        )
        state.setdefault("debug", {})["agent_messages"] = len(msgs)
        return state

    return node


# ======================================================================================
# 4) ROLE PROMPTS
# --------------------------------------------------------------------------------------
# Keep instructions crisp. We explicitly tell each agent to ask ONE question when a required
# input is missing. This, combined with preflight + interrupts, yields a robust clarify→resume loop.
# ======================================================================================

TRAINING_PROMPT = """You are a pragmatic Strength & Conditioning coach.
- If building plans, call build_workout().
- If evaluating recovery, call readiness_flag().
- Keep answers concise and actionable. If a required input is missing, ask ONE short question.
"""

NUTRITION_PROMPT = """You are a precision Nutrition coach.
- Compute energy first with tdee() if inputs present. For macro targets, call macro_plan().
- REQUIRED for macro_plan(): calories, weight_kg, protein_g_per_kg. Do NOT assume defaults; ask if missing.
- Keep it structured, include kcal/g units. If a required input is missing, ask ONE short question.
"""

BIOHACKING_PROMPT = """You are a cautious Biohacking coach (evidence-first).
- Focus on sleep/light/exercise timing, HRV, caffeine/creatine/VitD basics.
- Use readiness_flag() when HRV+RHR are given. Keep it practical; avoid medical claims.
- If a required input is missing, ask ONE short question.
"""


# ======================================================================================
# 5) SUPERVISOR + ROUTERS
# --------------------------------------------------------------------------------------
# We use a *supervisor* to choose one next agent based on the user's message.
# After an agent finishes, the *post_router* may route to another agent (to cover
# multi-domain requests) or decide we're "done" (go to human review).
# We keep both routers narrow and deterministic by using Pydantic schemas.
# ======================================================================================

class RouteDecision(BaseModel):
    route: Literal["training", "nutrition", "biohacking", "done"] = Field(..., description="Which agent?")
    reason: Optional[str] = None  # used for telemetry/debug; not displayed to user


ROUTER_PROMPT = """You are the Supervisor Router.
Agents: ["training","nutrition","biohacking"].
Return ONLY JSON for schema: {route: 'training'|'nutrition'|'biohacking'|'done', reason?: str}.
- 'training' for workouts, programming, recovery readiness via HRV/RHR.
- 'nutrition' for calories/macros/diets/meal planning.
- 'biohacking' for lifestyle levers (sleep, light, HRV, cold/heat, caffeine timing).
- 'done' if none fit.
"""

# `with_structured_output` ensures we either get a valid RouteDecision or raise.
router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(RouteDecision)


def supervisor(state: AppState) -> AppState:
    """
    First-hop router. Reads the *latest* user input and chooses exactly one agent or 'done'.
    """
    _init_memory(state)
    msgs = [SystemMessage(content=ROUTER_PROMPT), HumanMessage(content=f"User message: {state['user_input']}")]
    try:
        decision: RouteDecision = router_llm.invoke(msgs)
        route = decision.route if decision.route in ("training", "nutrition", "biohacking", "done") else "done"
        state.setdefault("debug", {})["router_reason"] = decision.reason
    except Exception as e:
        route = "done"
        state.setdefault("debug", {})["router_error"] = str(e)
    state["route"] = route
    return state


def decide_route(state: AppState) -> str:
    """
    Small adapter required by LangGraph to resolve the outgoing edge name from a node.
    """
    return state["route"]


POST_ROUTER_PROMPT = """You are the Post-Run Router.
Decide whether to route to ANOTHER agent or finish.
Return ONLY JSON: {route: 'training'|'nutrition'|'biohacking'|'done', reason?: str}.
Guidelines:
- If user's message spans multiple domains, pick the next most-relevant agent different from the previous one.
- Otherwise choose 'done'.
- Prefer not to repeat the same agent unless essential new inputs emerged.
"""

post_router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(RouteDecision)


def post_router(state: AppState) -> AppState:
    """
    Second router that runs *after* an agent finishes.
    It may route to a different agent for multi-domain requests (bounded by hop/visit limits),
    or route to 'done' to end the run at the human review gate.
    """
    _init_memory(state)

    # Hard stops: disable multi-hop or exceeded hop limit
    if not state.get("allow_multi", True) or state["hops"] >= state.get("max_hops", 3):
        state["route"] = "done"
        return state

    last = state["route_history"][-1] if state["route_history"] else None
    counts = state.get("agent_visit_counts", {})

    msgs = [
        SystemMessage(content=POST_ROUTER_PROMPT),
        HumanMessage(
            content=(
                f"User message: {state['user_input']}\n"
                f"Last agent: {last}\n"
                f"Route history: {state.get('route_history')}\n"
                f"Visit counts: {counts}\n"
            )
        ),
    ]
    try:
        decision: RouteDecision = post_router_llm.invoke(msgs)
        route = decision.route if decision.route in ("training", "nutrition", "biohacking", "done") else "done"

        # Guard against too many repeats of the same agent in a row.
        if route != "done" and route == last and counts.get(route, 0) >= state.get("repeat_limit", 2):
            route = "done"

        state.setdefault("debug", {})["post_router_reason"] = decision.reason
    except Exception as e:
        route = "done"
        state.setdefault("debug", {})["post_router_error"] = str(e)

    state["route"] = route
    return state


# ======================================================================================
# 6) HUMAN REVIEW GATE (dynamic interrupt)
# --------------------------------------------------------------------------------------
# This node always pauses and asks for approval. You can change it to only ask on risky
# outputs or above a size threshold. When the run is resumed with "yes"/"no", we finalize.
# ======================================================================================

def human_review(state: AppState) -> AppState:
    """
    Pause to ask a human to approve/reject the final plan. The *return value* of `interrupt`
    is injected when resuming with Command(resume=...). We then stamp the verdict into `result`.
    """
    preview = (state.get("result") or "")[:800]  # keep preview compact; UI can show full
    answer = interrupt(
        {
            "type": "review",
            "question": "Approve this plan? (yes/no)",
            "preview": preview,
            "route_history": state.get("route_history", []),
            "visit_counts": state.get("agent_visit_counts", {}),
            "hops": state.get("hops", 0),
        }
    )
    verdict = (str(answer) if answer is not None else "").strip().lower()
    if verdict in ("yes", "y", "approve", "approved", "ok", "go"):
        state["result"] = f"✅ Approved by human.\n\n{state['result']}"
    else:
        state["result"] = f"🚫 Rejected by human: '{answer}'. Consider revising specifics."
    return state


# ======================================================================================
# 7) BUILD GRAPH
# --------------------------------------------------------------------------------------
# Topology:
#   supervisor → (training|nutrition|biohacking|human_review)
#   agent      → post_router
#   post_router → (next agent | human_review)
#   human_review → END
#
# Every agent hop is traced, and multi-hop is bounded by `max_hops` and `repeat_limit`.
# ======================================================================================

training_agent = make_agent("training", TRAINING_PROMPT, [build_workout, readiness_flag])
nutrition_agent = make_agent("nutrition", NUTRITION_PROMPT, [tdee, macro_plan])
biohacking_agent = make_agent("biohacking", BIOHACKING_PROMPT, [readiness_flag])

graph = StateGraph(AppState)
graph.set_entry_point("supervisor")

graph.add_node("supervisor", supervisor)
graph.add_node("training", training_agent)
graph.add_node("nutrition", nutrition_agent)
graph.add_node("biohacking", biohacking_agent)
graph.add_node("post_router", post_router)
graph.add_node("human_review", human_review)

# First hop: supervisor chooses exactly one agent (or "done" → review)
graph.add_conditional_edges(
    "supervisor",
    decide_route,
    {"training": "training", "nutrition": "nutrition", "biohacking": "biohacking", "done": "human_review"},
)

# After ANY agent, decide next hop (or finish)
graph.add_edge("training", "post_router")
graph.add_edge("nutrition", "post_router")
graph.add_edge("biohacking", "post_router")

graph.add_conditional_edges(
    "post_router",
    decide_route,
    {"training": "training", "nutrition": "nutrition", "biohacking": "biohacking", "done": "human_review"},
)

graph.add_edge("human_review", END)


# ======================================================================================
# 8) COMPILE WITH CHECKPOINTER
# --------------------------------------------------------------------------------------
# A checkpointer is required for interrupts so LangGraph can persist/restore execution.
# MemorySaver keeps state in-memory for the life of the process. For durability across
# restarts, switch to SqliteSaver or Redis.
# ======================================================================================

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# TIP: To pause automatically at fixed points (without writing `interrupt`), you can use:
# app = graph.compile(
#     checkpointer=checkpointer,
#     interrupt_before=["human_review"],                 # static pause before review
#     interrupt_after=["training", "nutrition", "biohacking"],  # static pause after agents
# )


# ======================================================================================
# 9) CLI DEMO: run → catch interrupts → resume with Command(resume=...)
# --------------------------------------------------------------------------------------
# This controller emulates a chat loop:
#   - Start a run with `app.invoke(...)` and a thread_id (conversation id).
#   - If an interrupt occurs, LangGraph returns a state containing "__interrupt__".
#   - Read the interrupt payload (question/preview), collect a human answer,
#     then resume with `app.invoke(Command(resume=<answer>), config=thread)`.
#
# Notes:
#   - Keep the SAME `thread_id` for every resume. That's how LangGraph knows which paused
#     execution to continue.
#   - In a server, you'd replace input() with your UI and store thread_id per user/session.
#   - For streaming UIs, switch to `app.stream(...)` and listen for "interrupt" events.
# ======================================================================================

if __name__ == "__main__":
    thread = {"configurable": {"thread_id": "demo-thread-command"}}

    # Step 1: start a run (covers nutrition + training; might still ask clarifying Qs)
    state = app.invoke(
        {
            "user_input": (
                "Male, 82kg, 180cm, 29y, moderate activity. "
                "I want a 2200 kcal fat-loss macro plan and a simple 4-day workout."
            )
        },
        config=thread,
    )

    # ---- Main interrupt loop ----
    while state.get("__interrupt__"):
        payload = _interrupt_payload(state)

        print("\n[INTERRUPT]")
        print("Type:", payload.get("type"))
        print("Question:", payload.get("question"))
        if "preview" in payload:
            print("Preview:", (payload.get("preview") or "")[:240], "...")

        # Choose a sensible default based on the question text (CLI convenience only).
        # In production: present a form/UI with the exact required fields.
        q = (payload.get("question") or "").lower()
        if payload.get("type") == "review":
            default = "approve"
        elif "protein" in q or "g/kg" in q:
            default = "protein 2.0 g/kg"
        elif "experience" in q:
            default = "intermediate"
        elif "days per week" in q or "days/week" in q or "how many days" in q:
            default = "4"
        elif "goal" in q:
            default = "fat_loss"
        elif "hrv" in q:
            default = "52"
        elif "resting hr" in q or "rhr" in q:
            default = "62"
        else:
            default = "yes"  # generic fallback for Y/N prompts

        try:
            answer = input("Your answer: ").strip() or default
        except EOFError:
            # If stdin isn't interactive (e.g., CI), pick the default.
            answer = default

        # Resume exactly where it paused (must use the same thread_id!)
        state = app.invoke(Command(resume=answer), config=thread)

    # ---- End of run: show result + traces ----
    print("\n=== FINAL RESULT ===\n", state.get("result", ""))
    print("\n[ROUTE HISTORY] ", state.get("route_history"))
    print("[VISIT COUNTS]  ", state.get("agent_visit_counts"))
    print("[HOPS]          ", state.get("hops"))
    print("\n--- TRACE LOG ---")
    for i, t in enumerate(state.get("traces", []), 1):
        print(f"\n[{i}] Agent: {t['agent']}")
        print("Tools used:", [x.get("name") for x in t["tools_used"]])
        if t["tools_used"]:
            print(" First tool call:", t["tools_used"][0])
        print(" Output preview:", (t["output_preview"] or "")[:240], "...")

