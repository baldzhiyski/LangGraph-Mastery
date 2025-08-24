"""
Multi-Query RAG with Fusion (RRF) for a PDF knowledge base using:
- LangChain (LLM, embeddings, loaders, retriever)
- Chroma (local vector DB)
- LangGraph (LLM + tool execution loop)

Flow:
User question -> generate 3 sub-queries -> retrieve per sub-query -> RRF fuse -> structured context -> LLM answer
"""

# ========================
# Imports & Setup
# ========================
from dotenv import load_dotenv
import os
import hashlib
from typing import TypedDict, Annotated, Sequence, List, Dict

from langgraph.graph import StateGraph, END
from operator import add as add_messages

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma  # ✅ fixed deprecation (was langchain.vectorstores.Chroma)


# ========================
# Environment & Models
# ========================
load_dotenv()  # expects OPENAI_API_KEY in your .env

# Deterministic LLM to minimize hallucinations
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Embedding model compatible with your retriever/vector DB
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# ========================
# Data Loading: PDF -> Chunks -> Chroma
# ========================
pdf_path = "Stock_Market_Performance_2024.pdf"

# Safety check
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

# Load the PDF into Document objects (one per page)
pdf_loader = PyPDFLoader(pdf_path)
try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise

# Split into overlapping chunks to preserve context
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,   # chars per chunk
    chunk_overlap=200  # overlap to avoid context loss across boundaries
)
pages_split = text_splitter.split_documents(pages)

# Prepare Chroma vector store (persisted locally)
persist_directory = r"C:\Vaibhav\LangGraph_Book\LangGraphCourse\Agents"
collection_name = "stock_market"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    # Create/overwrite a collection with fresh embeddings
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print("Created ChromaDB vector store!")
except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise

# Turn vectorstore into a retriever (top-k similarity search)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # retrieve 5 chunks per query
)


# ========================
# Multi-Query + Fusion Helpers
# ========================

def generate_subqueries(question: str, n: int = 3) -> List[str]:
    """
    Use the configured LLM (temperature=0) to produce N diverse, focused sub-queries.
    - Each sub-query targets a different angle: overview, metrics/dates, drivers/impacts/comparisons.
    - We parse a numbered list for robustness.
    """
    sys = SystemMessage(content=(
        "Rewrite the user's question into multiple diverse, specific retrieval queries. "
        "Each query must be distinct and cover a different angle (overview, metrics/dates, drivers/impacts, comparisons). "
        f"Return exactly {n} numbered lines, with just the queries and no extra text."
    ))
    usr = HumanMessage(content=f"User question:\n{question}\n\nWrite {n} diverse retrieval queries.")
    resp = llm.invoke([sys, usr])  # sync call using your GPT-4o config

    # Extract queries from lines like "1. ...", "2) ...", etc.
    queries: List[str] = []
    for line in (resp.content or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if line[0].isdigit():
            # Strip leading numbering token ("1.", "1)", "1 -", etc.)
            parts = line.split(" ", 1)
            q = parts[1] if len(parts) > 1 else line
            q = q.lstrip(").:-").strip()
            if q:
                queries.append(q)

    # Fallback if parsing yields fewer than n queries
    if len(queries) < n:
        queries += [question] * (n - len(queries))

    return queries[:n]


def _cite(doc) -> str:
    """
    Build a short inline citation like [PDF, page 3] from doc.metadata.
    """
    meta = getattr(doc, "metadata", {}) or {}
    src = meta.get("source", "PDF")
    page = meta.get("page", None)
    page_str = f", page {page+1}" if isinstance(page, int) else ""
    return f"[{src}{page_str}]"


def _trim(text: str, limit: int = 1000) -> str:
    """
    Avoid prompt bloat by trimming long chunks.
    """
    if not text:
        return ""
    t = text.strip().replace("\n", " ")
    return t if len(t) <= limit else t[:limit].rstrip() + "…"


def _doc_key(doc) -> str:
    """
    Fingerprint a document chunk for de-duplication during fusion.
    Uses (source, page, md5(content)) to be robust.
    """
    meta = getattr(doc, "metadata", {}) or {}
    src = str(meta.get("source", "PDF"))
    page = str(meta.get("page", ""))
    content = getattr(doc, "page_content", "") or ""
    digest = hashlib.md5(content.encode("utf-8")).hexdigest()
    return f"{src}::{page}::{digest}"


def rrf_merge(per_query_docs: Dict[str, List], rrf_k: int = 60) -> List:
    """
    Reciprocal Rank Fusion (RRF):
    - Input: {sub_query: [doc1, doc2, ...]} each list ordered by similarity (rank 1 is best)
    - Output: single, de-duplicated ranked list by fused score
    score(doc) = sum_over_queries( 1 / (rrf_k + rank_q(doc)) )
    """
    scores: Dict[str, float] = {}
    keeper: Dict[str, object] = {}

    for sq, docs in per_query_docs.items():
        for rank, d in enumerate(docs, start=1):
            key = _doc_key(d)
            if key not in scores:
                scores[key] = 0.0
                keeper[key] = d
            scores[key] += 1.0 / (rrf_k + rank)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [keeper[key] for key, _ in ranked]


def format_fused_context(
    question: str,
    subqueries: List[str],
    per_query_docs: Dict[str, List],
    merged_docs: List,
    top_k_merged: int = 8,
    per_query_k: int = 5,
    snippet_limit: int = 800,
    include_per_query_sections: bool = True,
) -> str:
    """
    Build a structured context for the LLM:
    - USER QUESTION
    - TOP EVIDENCE (RRF-Fused): concise shortlist for grounding
    - (Optional) EVIDENCE BY SUB-QUERY: keeps transparency & debuggability
    """
    # Fused shortlist
    merged_lines: List[str] = []
    for i, d in enumerate(merged_docs[:top_k_merged], start=1):
        merged_lines.append(f"• {i}. {_cite(d)}\n{_trim(d.page_content, snippet_limit)}")

    # Per-subquery sections
    per_sections: List[str] = []
    if include_per_query_sections:
        per_sections.append("## EVIDENCE BY SUB-QUERY")
        for sq in subqueries:
            docs = per_query_docs.get(sq, []) or []
            per_sections.append(f"### Sub-Query: {sq}")
            if not docs:
                per_sections.append("(no results)")
                continue
            for i, d in enumerate(docs[:per_query_k], start=1):
                per_sections.append(f"- {i}. {_cite(d)}\n{_trim(d.page_content, snippet_limit)}")

    return (
        "## USER QUESTION\n"
        f"{question}\n\n"
        "## TOP EVIDENCE (RRF-Fused)\n"
        + ("\n".join(merged_lines) if merged_lines else "(no fused results)")
        + ("\n\n" + "\n".join(per_sections) if include_per_query_sections else "")
    )


# ========================
# Tool: Multi-Query Retrieval with RRF Fusion (with console debug)
# ========================
@tool
def multi_retriever_fusion_tool(query: str) -> str:
    """
    Expand the user's query into 3 sub-queries, retrieve top-k docs per sub-query,
    fuse with Reciprocal Rank Fusion (RRF), and return a structured context block.

    Console debug:
    - Prints the generated sub-queries
    - Prints retrieved chunks per sub-query (source/page + short snippet)
    """
    # 1) Create diverse sub-queries
    subqueries = generate_subqueries(query, n=3)

    # 🔍 Debug print: show the generated sub-queries
    print("\n--- Generated Sub-Queries ---")
    for i, sq in enumerate(subqueries, start=1):
        print(f"{i}. {sq}")
    print("-----------------------------")

    # 2) Retrieve independently for each sub-query
    per_query_docs: Dict[str, List] = {}
    k_per_query = 5  # keep aligned with your retriever

    print("\n--- Retrieval Results (per Sub-Query) ---")
    for sq in subqueries:
        docs = retriever.invoke(sq)  # ranked docs per sub-query (no scores exposed by retriever)
        per_query_docs[sq] = docs

        print(f"\nSub-Query: {sq}")
        if not docs:
            print("  (no results)")
            continue

        for idx, d in enumerate(docs[:k_per_query], start=1):
            print(f"  {idx}. {_cite(d)}")
            print(f"     {_trim(d.page_content, limit=220)}")
    print("\n--- End Retrieval Results ---\n")

    # 3) Fuse results (deduplicate + consensus ranking)
    merged_docs = rrf_merge(per_query_docs, rrf_k=60)

    # 4) Build context for the LLM (compact + transparent)
    context_block = format_fused_context(
        question=query,
        subqueries=subqueries,
        per_query_docs=per_query_docs,
        merged_docs=merged_docs,
        top_k_merged=8,           # fused shortlist size
        per_query_k=5,            # how many to show per sub-query (for transparency)
        snippet_limit=800,        # trim long chunks
        include_per_query_sections=True
    )
    return context_block


# ========================
# LangGraph: State, Nodes, Edges
# ========================
class AgentState(TypedDict):
    # Conversation state is a list of LC messages; `add_messages` appends new ones
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState):
    """
    If the last LLM output contains tool calls, the graph should continue to the tool node.
    """
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


# System prompt guides the assistant to use our tool + cite sources
system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the loaded PDF.
Use `multi_retriever_fusion_tool` to:
1) Expand the user's query into 3 sub-queries,
2) Retrieve evidence for each,
3) Use the RRF-fused 'Top Evidence' as primary grounding, and optionally consult per-sub-query sections.

Important: Call `multi_retriever_fusion_tool` at most once per user question.

When answering:
- Be concise, factual, and reference the evidence.
- Cite sources inline with the bracketed tags included in the context (e.g., [PDF, page 5]).
- If information is missing or contradictory, say so explicitly.
"""


# Bind tool(s) to the LLM so it can call them autonomously
tools = [multi_retriever_fusion_tool]
llm = llm.bind_tools(tools)

# Map tool name to callable for execution node
tools_dict = {our_tool.name: our_tool for our_tool in tools}


def call_llm(state: AgentState) -> AgentState:
    """
    LLM node:
    - Prepend system prompt
    - Ask the LLM for either an answer or tool calls
    """
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {'messages': [message]}


def take_action(state: AgentState) -> AgentState:
    """
    Tool execution node:
    - Execute any tool calls from the last LLM message
    - Return ToolMessage results so the LLM can continue/reason with them

    Guard: enforce at most ONE tool execution per turn (helps avoid duplicate calls).
    """
    tool_calls = state['messages'][-1].tool_calls
    results = []

    max_tool_calls_per_turn = 1
    for idx, t in enumerate(tool_calls):
        if idx >= max_tool_calls_per_turn:
            print("Skipping extra tool calls this turn (limited to 1).")
            results.append(
                ToolMessage(
                    tool_call_id=t['id'],
                    name=t['name'],
                    content="Skipped: Only one tool call is allowed per turn."
                )
            )
            continue

        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        if t['name'] not in tools_dict:
            # Defensive: unknown tool name
            result = "Incorrect Tool Name, Please Retry and Select tool from Available tools."
        else:
            # Execute tool and capture string output
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")

        # Feed the tool's output back into the graph as a ToolMessage
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}


# Build the graph
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)                 # LLM reasoning node
graph.add_node("retriever_agent", take_action)  # Tool execution node

# If LLM requests a tool -> go to retriever_agent; else end
graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)

# After running tools, return to the LLM for synthesis
graph.add_edge("retriever_agent", "llm")

# Entry point
graph.set_entry_point("llm")

# Compile into a runnable app
rag_agent = graph.compile()


# ========================
# CLI Runner
# ========================
def running_agent():
    """
    Simple REPL loop:
    - Reads user input
    - Invokes the graph with a HumanMessage
    - Prints the LLM's final response
    """
    print("\n=== RAG AGENT (Multi-Query + RRF Fusion) ===")
    print("Type 'exit' or 'quit' to end.")

    while True:
        try:
            user_input = input("\nWhat is your question: ")
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        if user_input.lower() in ['exit', 'quit']:
            break

        # Wrap user input into a HumanMessage and run the graph
        messages = [HumanMessage(content=user_input)]
        result = rag_agent.invoke({"messages": messages})

        # The last message is the latest model output
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)


# Run the REPL if executed as a script
if __name__ == "__main__":
    running_agent()
