"""그래프가 실제로 컴파일되는지, 컨테이너가 부를 진입점이 맞는지 본다.

노드 이름과 분기 대상이 어긋나면 컴파일에서 잡히고, uvicorn 이 부르는
`app.main:app` 이 사라지면 두 번째 테스트에서 잡힌다.
"""

from langgraph.checkpoint.memory import InMemorySaver

from app.chat.graph import build_graph

NODES = {"analyze", "chat", "summarize", "ask_change", "search", "wishlist"}


def test_graph_compiles_with_every_node():
    graph = build_graph(InMemorySaver())
    assert NODES <= set(graph.get_graph().nodes)


def test_asgi_app_exposes_the_documented_paths():
    from app.main import app

    # OpenAPI 문서가 Backend 가 실제로 보게 되는 표면이다.
    paths = app.openapi()["paths"]
    assert "post" in paths["/api/v1/chat/stream"]
    assert "delete" in paths["/api/v1/chat/{chat_id}"]
