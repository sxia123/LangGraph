import operator
import time
from typing import Annotated, Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from src.core.local_llm import LocalLLMClient


class CodeReviewState(TypedDict):
    messages: Annotated[List[Dict[str, Any]], operator.add]
    task: str
    code: str
    review: str
    approved: bool
    revision_count: int


def create_code_review_team_graph(llm_client: LocalLLMClient):
    workflow = StateGraph(CodeReviewState)

    def developer_node(state: CodeReviewState) -> Dict[str, Any]:
        existing_code = state.get("code", "")
        if existing_code and state.get("revision_count", 0) == 0:
            return {"revision_count": 1}

        prompt = f"""You are Lead Software Developer. Task: "{state.get("task", "")}"."""
        if existing_code:
            prompt += f"\nExisting Code to revise:\n{existing_code}\nReviewer Feedback: {state.get('review', '')}"

        res = llm_client.generate_completion(prompt, state.get("messages", []))

        msg = {
            "id": f"dev_{int(time.time() * 1000)}",
            "sender": "Developer Node",
            "role": "assistant",
            "content": res.content,
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "code": res.content,
            "messages": [msg],
            "revision_count": state.get("revision_count", 0) + 1,
        }

    def reviewer_node(state: CodeReviewState) -> Dict[str, Any]:
        prompt = f"""You are Code Auditor. Review code:\n{state.get("code", "")}\nOutput APPROVED if valid."""
        res = llm_client.generate_completion(prompt, state.get("messages", []))
        is_approved = "APPROVED" in res.content.upper()

        msg = {
            "id": f"rev_{int(time.time() * 1000)}",
            "sender": "Reviewer Node",
            "role": "assistant",
            "content": res.content,
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {"review": res.content, "approved": is_approved, "messages": [msg]}

    def route_review(state: CodeReviewState) -> str:
        if state.get("approved") or state.get("revision_count", 0) >= 3:
            return END
        return "developer"

    workflow.add_node("developer", developer_node)
    workflow.add_node("reviewer", reviewer_node)

    workflow.add_edge(START, "developer")
    workflow.add_edge("developer", "reviewer")
    workflow.add_conditional_edges("reviewer", route_review, {"developer": "developer", END: END})

    return workflow.compile()


# Default compiled graph instance for LangGraph Studio CLI
default_code_review_graph = create_code_review_team_graph(LocalLLMClient())
