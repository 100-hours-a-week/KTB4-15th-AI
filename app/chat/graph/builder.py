"""그래프 조립.

턴마다 START부터 다시 돈다. 턴을 넘기는 장치는 interrupt가 아니라
State의 awaiting_confirm 플래그다.

입구에서 source_type으로 한 번 갈라진다. 찜 기반 추천은 그 턴에 추천만 하고 끝나며,
다음 턴부터는 평소대로 analyze를 거친다.
"""

from langgraph.graph import END, START, StateGraph

from app.chat.graph import nodes
from app.chat.graph.routing import ENTRY_ROUTES, ROUTES, entry_route, route
from app.chat.graph.state import ChatState


def build_graph(checkpointer):
    graph = StateGraph(ChatState)

    graph.add_node("analyze", nodes.analyze)
    graph.add_node("chat", nodes.chat)
    graph.add_node("summarize", nodes.summarize)
    graph.add_node("ask_change", nodes.ask_change)
    graph.add_node("search", nodes.search)
    graph.add_node("wishlist", nodes.wishlist)

    graph.add_conditional_edges(START, entry_route, ENTRY_ROUTES)
    graph.add_conditional_edges("analyze", route, ROUTES)

    for node in ("chat", "summarize", "ask_change", "search", "wishlist"):
        graph.add_edge(node, END)

    return graph.compile(checkpointer=checkpointer)
