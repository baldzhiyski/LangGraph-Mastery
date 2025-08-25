from typing import TypedDict
from langgraph.graph import StateGraph,START,END

class PortfolioState(TypedDict):
    amount_usd:float
    total_usd:float
    total_inr:float

 
def calc_total_inr(state:PortfolioState) -> PortfolioState:
    state['total_inr'] = state['amount_usd'] *85
    return state

def calc_total(state:PortfolioState) -> PortfolioState:
    state['total_usd'] = state['amount_usd'] *1.08
    return state


builder = StateGraph(PortfolioState)

builder.add_node('calc_total_node',calc_total)
builder.add_node('calc_total_inr',calc_total_inr)

builder.add_edge(START,'calc_total_node')
builder.add_edge('calc_total_node','calc_total_inr')
builder.add_edge('calc_total_inr',END)

app = builder.compile()

result = app.invoke({'amount_usd':1000})
print(result)
