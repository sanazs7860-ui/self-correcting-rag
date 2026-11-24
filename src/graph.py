from langgraph.graph import StateGraph, END
from .state import GraphState
from . import nodes as n


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("retrieve", n.retrieve)
    g.add_node("grade_documents", n.grade_documents)
    g.add_node("rewrite", n.rewrite_query)
    g.add_node("web_search", n.web_search)
    g.add_node("generate", n.generate)
    g.add_node("check_answer", n.check_answer)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges("grade_documents", n.decide_after_grading,
                            {"generate": "generate", "rewrite": "rewrite", "web_search": "web_search"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("web_search", "generate")
    g.add_edge("generate", "check_answer")
    g.add_conditional_edges("check_answer", n.decide_after_check,
                            {"generate": "generate", "rewrite": "rewrite", "end": END})
    return g.compile()


def initial_state(question: str, max_retries: int = 1, verify: bool = True) -> dict:
    return {"original_question": question, "question": question, "documents": [],
            "generation": "", "retries": 0, "gen_retries": 0, "max_retries": max_retries,
            "verify": verify, "used_web": False, "grounded": False, "useful": False, "trace": []}
