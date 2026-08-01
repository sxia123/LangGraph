import operator
import time
from typing import Annotated, Any, Dict, List

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.core.local_llm import LocalLLMClient


class SolutionReviewState(TypedDict, total=False):
    messages: Annotated[List[Dict[str, Any]], operator.add]
    task: str
    solution: str
    review: str
    approved: bool
    revision_count: int


def _get_task(state: SolutionReviewState) -> str:
    task = state.get("task", "")
    if not task:
        messages = state.get("messages") or []
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, dict):
                task = last_msg.get("content", "")
            elif hasattr(last_msg, "content"):
                task = getattr(last_msg, "content", "")
    return task


def create_solution_review_team_graph(llm_client: LocalLLMClient):
    workflow = StateGraph(SolutionReviewState)

    def specialist_node(state: SolutionReviewState) -> Dict[str, Any]:
        existing_solution = state.get("solution", "")
        if existing_solution and state.get("revision_count", 0) == 0:
            return {"revision_count": 1}

        task = _get_task(state)
        prompt = f"""You are Lead Solution Specialist. Task: "{task}"."""
        if existing_solution:
            prompt += f"\nExisting Solution to revise:\n{existing_solution}\nReviewer Feedback: {state.get('review', '')}"

        res = llm_client.generate_completion(prompt, state.get("messages", []))

        msg = {
            "id": f"spec_{int(time.time() * 1000)}",
            "sender": "Specialist Node",
            "role": "assistant",
            "content": res.content,
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "solution": res.content,
            "messages": [msg],
            "revision_count": state.get("revision_count", 0) + 1,
        }

    def reviewer_node(state: SolutionReviewState) -> Dict[str, Any]:
        prompt = f"""You are Quality Auditor. Review solution:\n{state.get("solution", "")}\nOutput APPROVED if valid."""
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

    def route_review(state: SolutionReviewState) -> str:
        if state.get("approved") or state.get("revision_count", 0) >= 3:
            return END
        return "specialist"

    workflow.add_node("specialist", specialist_node)
    workflow.add_node("reviewer", reviewer_node)

    workflow.add_edge(START, "specialist")
    workflow.add_edge("specialist", "reviewer")
    workflow.add_conditional_edges("reviewer", route_review, {"specialist": "specialist", END: END})

    return workflow.compile()


# Default compiled graph instance for LangGraph Studio CLI
default_solution_review_graph = create_solution_review_team_graph(LocalLLMClient())
