from typing import Literal, TypedDict
from langgraph.graph import StateGraph,START,END

class PortfolioState(TypedDict):
    amount_usd:float
    total_usd:float
    target_currency : Literal["INR","EUR"]
    total_inr:float
    total:float

def calc_total_inr(state:PortfolioState) -> PortfolioState:
    state['total_inr'] = state['total_usd'] *85
    return state

def calc_total(state:PortfolioState) -> PortfolioState:
    state['total_usd'] = state['amount_usd'] *1.08
    return state

def calc_total_eur(state:PortfolioState) -> PortfolioState:
    state['total'] = state['total_usd'] *0.9
    return state

def choose_conversion(state:PortfolioState) -> PortfolioState:
    return state['target_currency']

builder = StateGraph(PortfolioState)

builder.add_node('calc_total_node',calc_total)
builder.add_node('calc_total_inr',calc_total_inr)
builder.add_node('calc_total_eur',calc_total_eur)

builder.add_edge(START,'calc_total_node')
builder.add_conditional_edges(
    "calc_total_node",
    choose_conversion,
    {
        "INR":"calc_total_inr",
        "EUR":"calc_total_eur"
    }
)
builder.add_edge(["calc_total_inr","calc_total_eur"],END)

app = builder.compile()

result = app.invoke({'amount_usd':1000,'target_currency':'EUR'})
print(result)