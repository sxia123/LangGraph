import operator
import time
from typing import Annotated, Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from src.core.local_llm import LocalLLMClient


class MultiAgentState(TypedDict):
    messages: Annotated[List[Dict[str, Any]], operator.add]
    current_task: str
    next_agent: str
    research_output: str
    coder_output: str
    critic_feedback: str
    final_response: str
    agent_thoughts: Annotated[List[Dict[str, Any]], operator.add]


def create_multi_agent_supervisor_graph(llm_client: LocalLLMClient):
    workflow = StateGraph(MultiAgentState)

    # 1. SUPERVISOR ROUTER NODE
    def supervisor_node(state: MultiAgentState) -> Dict[str, Any]:
        prompt = f"""You are the Multi-Agent Supervisor Router for a team of AI worker agents:
- 'researcher': Researches facts, web search, background data.
- 'coder': Writes code, scripts, or technical functions.
- 'critic': Audits code/research output for bugs, security risks, or missing edge cases.
- 'writer': Synthesizes the final user response when all work is verified.
- 'FINISH': Stop when the user task is fully answered.

Task: "{state.get("current_task", "")}"
State Summary:
- Research Done: {"Yes" if state.get("research_output") else "No"}
- Code Generated: {"Yes" if state.get("coder_output") else "No"}
- Critic Approved: {"Yes" if state.get("critic_feedback") else "No"}

Respond with ONLY ONE word representing the next agent node: "researcher", "coder", "critic", "writer", or "FINISH"."""

        res = llm_client.generate_completion(prompt, state.get("messages", []))
        decision = res.content.strip().lower()

        target = "writer"
        if "researcher" in decision:
            target = "researcher"
        elif "coder" in decision:
            target = "coder"
        elif "critic" in decision:
            target = "critic"
        elif "finish" in decision or "writer" in decision:
            target = "writer"
        else:
            task = state.get("current_task", "").lower()
            if not state.get("research_output") and ("find" in task or "research" in task):
                target = "researcher"
            elif not state.get("coder_output"):
                target = "coder"
            elif not state.get("critic_feedback"):
                target = "critic"

        thought = (
            res.thought
            or f"Supervisor evaluated state and routed control to [{target.upper()}] node."
        )

        return {
            "next_agent": target,
            "agent_thoughts": [
                {"agent": "Supervisor", "thought": thought, "timestamp": time.strftime("%H:%M:%S")}
            ],
        }

    # 2. RESEARCHER NODE
    def researcher_node(state: MultiAgentState) -> Dict[str, Any]:
        prompt = f"""You are the Expert Researcher Agent. Gather findings for: "{state.get("current_task", "")}"."""
        res = llm_client.generate_completion(
            prompt, state.get("messages", []), available_tools=["web_search"]
        )

        msg = {
            "id": f"msg_{int(time.time() * 1000)}",
            "sender": "Researcher Agent",
            "role": "assistant",
            "content": res.content,
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "research_output": res.content,
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Researcher",
                    "thought": res.thought or "Gathered domain research context.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 3. CODER NODE
    def coder_node(state: MultiAgentState) -> Dict[str, Any]:
        prompt = f"""You are the Lead Software Engineer Node.
Write clean, maintainable code for: "{state.get("current_task", "")}".
Research Context: {state.get("research_output", "N/A")}
Critic Feedback: {state.get("critic_feedback", "None")}"""

        res = llm_client.generate_completion(prompt, state.get("messages", []))
        msg = {
            "id": f"msg_{int(time.time() * 1000)}",
            "sender": "Coder Agent",
            "role": "assistant",
            "content": res.content,
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "coder_output": res.content,
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Coder",
                    "thought": res.thought or "Generated code solution.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 4. CRITIC NODE
    def critic_node(state: MultiAgentState) -> Dict[str, Any]:
        prompt = f"""You are the Senior QA Critic Node. Audit this code:\n{state.get("coder_output", "")}\nProvide verdict (APPROVED or REVISION)."""
        res = llm_client.generate_completion(prompt, state.get("messages", []))

        msg = {
            "id": f"msg_{int(time.time() * 1000)}",
            "sender": "Critic Agent",
            "role": "assistant",
            "content": res.content,
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "critic_feedback": res.content,
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Critic",
                    "thought": res.thought or "Audited solution code.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 5. WRITER NODE
    def writer_node(state: MultiAgentState) -> Dict[str, Any]:
        prompt = f"""You are the Final Synthesizer Agent. Consolidate final answer for: "{state.get("current_task", "")}"."""
        res = llm_client.generate_completion(prompt, state.get("messages", []))

        msg = {
            "id": f"msg_{int(time.time() * 1000)}",
            "sender": "Writer Agent",
            "role": "assistant",
            "content": res.content,
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "final_response": res.content,
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Writer",
                    "thought": res.thought or "Synthesized final deliverable.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # ROUTER CONDITIONAL FUNCTION
    def route_supervisor(state: MultiAgentState) -> str:
        next_agent = state.get("next_agent", "writer")
        if next_agent in ["researcher", "coder", "critic", "writer"]:
            return next_agent
        return END

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("writer", writer_node)

    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "researcher": "researcher",
            "coder": "coder",
            "critic": "critic",
            "writer": "writer",
            END: END,
        },
    )

    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("coder", "supervisor")
    workflow.add_edge("critic", "supervisor")
    workflow.add_edge("writer", END)

    return workflow.compile()


# Default compiled graph instance for LangGraph Studio CLI
default_supervisor_graph = create_multi_agent_supervisor_graph(LocalLLMClient())
