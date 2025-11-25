from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from receipt_agent.utils.nodes import (
    analyze_image_node,
    ask_split_node,
    calculate_itemization_node,
    get_split_instructions_node,
    load_image_node,
)
from receipt_agent.utils.state import ReceiptState

builder = StateGraph(ReceiptState)

# Add all nodes
builder.add_node("load", load_image_node)
builder.add_node("analyze", analyze_image_node)
builder.add_node("ask_split", ask_split_node)
builder.add_node("get_split_instructions", get_split_instructions_node)
builder.add_node("calculate_itemization", calculate_itemization_node)

# Add edges to define the flow
builder.add_edge(START, "load")
builder.add_edge("load", "analyze")
builder.add_edge("analyze", "ask_split")
builder.add_edge("ask_split", "get_split_instructions")
builder.add_edge("get_split_instructions", "calculate_itemization")
builder.add_edge("calculate_itemization", END)

graph = builder.compile()
