from dotenv import load_dotenv
from typing import Annotated,List
from langgraph.graph import  START,END,StateGraph
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from pydantic import BaseModel,Field

load_dotenv()

llm = init_chat_model("gtp-4o")

class State(TypedDict):
    messages: Annotated[list,add_messages]
    user_question: str | None
    google_results: str | None
    bing_results: str | None
    reddit_results: str | None
    selected_reddit_urls: list[str] | None
    reddit_post_data: list|None
    google_analysis: str|None
    reddit_analysis: str|None
    bing_analysis: str|None
    final_answer: str|None


def google_search(state: State):
    return

def bing_search(state: State):
    return

def reddit_search(state: State):
    return

def analyze_reddit_posts(state: State):
    return

def retrieve_reddit_posts(state: State):
    return

def analyze_google_results(state: State):
    return

def analyze_bing_results(state: State):
    return

def analyze_reddit_results(state: State):
    return

def synthesize_analysis(state: State):
    return

graph_builder = StateGraph(State)

graph_builder.add_node("google_search",google_search)
graph_builder.add_node("bing_search",bing_search)
graph_builder.add_node("reddit_search",reddit_search)
graph_builder.add_node("retrieve_reddit_posts",retrieve_reddit_posts)
graph_builder.add_node("analyze_reddit_posts",analyze_reddit_posts)
graph_builder.add_node("analyze_google_results",analyze_google_results)
graph_builder.add_node("analyze_bing_results",analyze_bing_results)
graph_builder.add_node("analyze_reddit_results",analyze_reddit_results)
graph_builder.add_node("synthesize_analysis",synthesize_analysis)


graph_builder.add_edge(START,"google_search")
graph_builder.add_edge(START,"bing_search")
graph_builder.add_edge(START,"reddit_search")

graph_builder.add_edge("google_search","analyze_reddit_posts")
graph_builder.add_edge("bing_search","analyze_reddit_posts")
graph_builder.add_edge("reddit_search","analyze_reddit_posts")
graph_builder.add_edge("analyze_reddit_posts","retrieve_reddit_posts")


graph_builder.add_edge("retrieve_reddit_posts","analyze_google_results")
graph_builder.add_edge("retrieve_reddit_posts","analyze_bing_results")
graph_builder.add_edge("retrieve_reddit_posts","analyze_reddit_results")

graph_builder.add_node("analyze_google_results","synthesize_analysis")
graph_builder.add_node("analyze_bing_results","synthesize_analysis")
graph_builder.add_node("analyze_reddit_results","synthesize_analysis")
graph_builder.add_node("synthesize_analysis",END)

graph = graph_builder.compile()

def run_chatbot():
    print("Multi-Source Research Agent")
    print("Type exit to quit\n")

    while True:
        user_input = input("Ask me anything: ")
        if user_input.lower() == "exit":
            print("Goodbye")
            break

        state = {
            "messages": [{"role": "user", "content" : user_input}],
            "user_question": user_input,
            "google_results": None,
            "bing_results": None,
            "reddit_results": None,
            "reddit_analysis": None,
            "google_analysis": None,
            "synthesize_analysis": None,
            "bing_analysis": None,
            "final_answer": None,
            "selected_reddit_urls": None,
            "reddit_post_data": None,
        }

        print("\n Starting parallel research process ...")
        print("Launching Google , Bing and Reddit searches ...\n")

        final_state = graph.invoke(state)

        if final_state.get("final_answer"):
            print(f"\nFinal answer\n:{final_state.get('final_answer')}\n")

        print("-" * 80)


if __name__ == "__main__":
    run_chatbot()