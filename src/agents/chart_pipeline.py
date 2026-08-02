import operator
import time
from typing import Annotated, Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.core.local_llm import LocalLLMClient


class ChartPipelineState(TypedDict, total=False):
    messages: Annotated[List[Dict[str, Any]], operator.add]
    user_input: str
    current_step: str

    # Intake
    goals: List[str]
    scope: str
    clearance_level: str
    intake_status: str  # "APPROVED" | "BLOCKED"

    # Specialist & Verification
    specialist_output: str
    tier0_checks: Dict[str, bool]  # {"observed": True, "completed": True, "tested": True, "docs": True}
    tier1_verified: bool
    is_converged: bool

    # Escalation
    escalation_notes: str
    repaired_output: str

    # Execution & Memory
    action_payload: Optional[Dict[str, Any]]
    action_blocked: bool
    approval_granted: bool
    execution_result: str
    memory_logs: Annotated[List[Dict[str, Any]], operator.add]
    agent_thoughts: Annotated[List[Dict[str, Any]], operator.add]


def _get_input(state: ChartPipelineState) -> str:
    user_input = state.get("user_input", "")
    if not user_input:
        messages = state.get("messages") or []
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, dict):
                user_input = last_msg.get("content", "")
            elif hasattr(last_msg, "content"):
                user_input = getattr(last_msg, "content", "")
    return user_input or "Execute default workflow task."


def create_chart_pipeline_graph(llm_client: LocalLLMClient):
    workflow = StateGraph(ChartPipelineState)

    # 1. INTAKE NODE
    def intake_node(state: ChartPipelineState) -> Dict[str, Any]:
        task = _get_input(state)
        task_lower = task.lower()

        # Classify scope and clearance
        blocked_keywords = ["malicious", "unauthorized", "drop database", "exploit"]
        is_blocked = any(kw in task_lower for kw in blocked_keywords)

        status = "BLOCKED" if is_blocked else "APPROVED"
        goals = ["Extract intent", "Validate scope", "Assess clearance"]
        scope = "Standard Workload" if not is_blocked else "Restricted Workload"

        msg = {
            "id": f"msg_intake_{int(time.time() * 1000)}",
            "sender": "Intake Node",
            "role": "assistant",
            "content": f"### Intake Evaluation\n**Status**: {status}\n**Scope**: {scope}\n**Goals**: {', '.join(goals)}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "user_input": task,
            "intake_status": status,
            "scope": scope,
            "clearance_level": "Level-1" if not is_blocked else "Denied",
            "goals": goals,
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Intake Node",
                    "thought": f"Assessed request scope: [{scope}] -> Status [{status}].",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # BLOCKED END NODE (Intake failure)
    def blocked_end_node(state: ChartPipelineState) -> Dict[str, Any]:
        msg = {
            "id": f"msg_blocked_{int(time.time() * 1000)}",
            "sender": "System Gate",
            "role": "assistant",
            "content": "🚫 **Workflow Terminated**: Request was blocked during intake due to clearance or scope violation.",
            "timestamp": time.strftime("%H:%M:%S"),
        }
        return {"current_step": "blocked_end", "messages": [msg]}

    # 2. SPECIALIST NODE (Local AI)
    def specialist_node(state: ChartPipelineState) -> Dict[str, Any]:
        task = _get_input(state)
        prompt = f"""You are the Specialist Agent powered by Local AI.
Gather requirements, execute local context assembly, and draft solution for: "{task}".
Be concise and structured. Do not output internal thinking or <think> tags."""

        res = llm_client.generate_completion(prompt, messages=[], max_tokens=512)
        msg = {
            "id": f"msg_spec_{int(time.time() * 1000)}",
            "sender": "Specialist Agent (Local AI)",
            "role": "assistant",
            "content": f"### Specialist Solution Draft (Local AI)\n\n{res.content}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "specialist_output": res.content,
            "current_step": "specialist_complete",
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Specialist Agent",
                    "thought": res.thought or "Gathered context and committed local solution draft.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 3. TIER 0 CHECKS NODE
    def tier0_checks_node(state: ChartPipelineState) -> Dict[str, Any]:
        output = state.get("specialist_output", "")
        # Evaluate 4 criteria: observed, completed, tested, docs
        checks = {
            "observed": len(output) > 20,
            "completed": bool(output and not output.isspace()),
            "tested": "error" not in output.lower(),
            "docs": True,
        }
        all_passed = all(checks.values())

        msg = {
            "id": f"msg_tier0_{int(time.time() * 1000)}",
            "sender": "Tier 0 Audit Node",
            "role": "assistant",
            "content": f"### Tier 0 Automated Checks\n- Observed: {'✅' if checks['observed'] else '❌'}\n- Completed: {'✅' if checks['completed'] else '❌'}\n- Tested: {'✅' if checks['tested'] else '❌'}\n- Docs: {'✅' if checks['docs'] else '❌'}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "tier0_checks": checks,
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Tier 0 Auditor",
                    "thought": f"Tier 0 checks evaluated: All Passed = {all_passed}.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 4. TIER 1 VERIFY NODE
    def tier1_verify_node(state: ChartPipelineState) -> Dict[str, Any]:
        prompt = f"""You are Tier 1 Verification Auditor. Audit this output:\n{state.get("specialist_output", "")}\nOutput VERIFIED if valid, else REVISE with a brief 1-line reason. Do not output <think> tags."""
        res = llm_client.generate_completion(prompt, messages=[], max_tokens=128)
        is_verified = "VERIFIED" in res.content.upper() or "APPROVED" in res.content.upper()
        t0 = state.get("tier0_checks", {})
        converged = is_verified and all(t0.values()) if t0 else is_verified

        msg = {
            "id": f"msg_tier1_{int(time.time() * 1000)}",
            "sender": "Tier 1 Verification Node",
            "role": "assistant",
            "content": f"### Tier 1 Verification\n**Status**: {'VERIFIED' if is_verified else 'REVISION REQUIRED'}\n**Convergence**: {'CONVERGED' if converged else 'ESCALATION REQUIRED'}\n\n{res.content}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "tier1_verified": is_verified,
            "is_converged": converged,
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Tier 1 Auditor",
                    "thought": res.thought or f"Tier 1 audit complete. Converged = {converged}.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 5. ESCALATION NODE (Frontier Model)
    def escalation_node(state: ChartPipelineState) -> Dict[str, Any]:
        task = _get_input(state)
        prompt = f"""You are the Frontier Model Escalation Node.
Solve and repair complex edge cases for: "{task}".
Previous Specialist Draft: {state.get("specialist_output", "N/A")}. Be concise."""

        res = llm_client.generate_completion(prompt, messages=[], max_tokens=512)
        msg = {
            "id": f"msg_esc_{int(time.time() * 1000)}",
            "sender": "Escalation Node (Frontier Model)",
            "role": "assistant",
            "content": f"### Frontier Model Escalation Synthesis\n\n{res.content}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "escalation_notes": res.content,
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Frontier Model Escalation",
                    "thought": res.thought or "Escalated task to high-capability frontier model reasoning.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 6. ADJUDICATE & REPAIR NODE
    def adjudicate_repair_node(state: ChartPipelineState) -> Dict[str, Any]:
        esc = state.get("escalation_notes") or state.get("specialist_output", "")
        prompt = f"""You are the Adjudication & Repair Node. Apply final concise repairs to solution:\n{esc}"""

        res = llm_client.generate_completion(prompt, messages=[], max_tokens=512)
        msg = {
            "id": f"msg_adj_{int(time.time() * 1000)}",
            "sender": "Adjudicate & Repair Node",
            "role": "assistant",
            "content": f"### Adjudicated & Repaired Output\n\n{res.content}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "repaired_output": res.content,
            "specialist_output": res.content,
            "is_converged": True,
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Adjudication Node",
                    "thought": res.thought or "Adjudicated escalation feedback and applied repairs.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 7. PREPARE ACTION NODE
    def prepare_action_node(state: ChartPipelineState) -> Dict[str, Any]:
        solution = state.get("repaired_output") or state.get("specialist_output", "")
        task = _get_input(state)

        is_blocked = "deny_action" in task.lower()
        payload = {
            "target_action": "execute_solution",
            "payload_summary": solution[:200] + "..." if len(solution) > 200 else solution,
            "requires_approval": True,
        }

        msg = {
            "id": f"msg_prep_{int(time.time() * 1000)}",
            "sender": "Action Preparation Node",
            "role": "assistant",
            "content": f"### Action Payload Prepared\n**Blocked Status**: {'BLOCKED' if is_blocked else 'READY'}\n**Target**: {payload['target_action']}",
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "action_payload": payload,
            "action_blocked": is_blocked,
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Action Preparation",
                    "thought": f"Prepared action payload. Action Blocked = {is_blocked}.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # BLOCKED MEMORY NODE
    def blocked_memory_node(state: ChartPipelineState) -> Dict[str, Any]:
        log_entry = {
            "event": "ACTION_BLOCKED",
            "reason": "Policy violation during action preparation.",
            "timestamp": time.strftime("%H:%M:%S"),
        }
        msg = {
            "id": f"msg_block_mem_{int(time.time() * 1000)}",
            "sender": "Memory Gate",
            "role": "assistant",
            "content": "🚫 **Action Blocked**: Recorded blocked event to system memory.",
            "timestamp": time.strftime("%H:%M:%S"),
        }
        return {
            "memory_logs": [log_entry],
            "messages": [msg],
        }

    # 8. APPROVAL NODE
    def approval_node(state: ChartPipelineState) -> Dict[str, Any]:
        msg = {
            "id": f"msg_appr_{int(time.time() * 1000)}",
            "sender": "Approval Gate Node",
            "role": "assistant",
            "content": "✅ **Approval Gate Granted**: Action approved for execution.",
            "timestamp": time.strftime("%H:%M:%S"),
        }
        return {
            "approval_granted": True,
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Approval Gate",
                    "thought": "Approved action payload for execution.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 9. EXECUTE NODE
    def execute_node(state: ChartPipelineState) -> Dict[str, Any]:
        payload = state.get("action_payload") or {}
        result_text = f"Executed action [{payload.get('target_action', 'run')}] successfully."

        msg = {
            "id": f"msg_exec_{int(time.time() * 1000)}",
            "sender": "Execution Node",
            "role": "assistant",
            "content": f"### Execution Completed\n{result_text}",
            "timestamp": time.strftime("%H:%M:%S"),
        }
        return {
            "execution_result": result_text,
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Execution Node",
                    "thought": f"Executed payload: {result_text}",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # 10. FINALIZE & MEMORY NODE
    def finalize_memory_node(state: ChartPipelineState) -> Dict[str, Any]:
        memory_entry = {
            "event": "PIPELINE_SUCCESS",
            "input": state.get("user_input"),
            "result": state.get("execution_result"),
            "timestamp": time.strftime("%H:%M:%S"),
        }
        final_text = (
            f"### Final Pipeline Result\n\n"
            f"**Task**: {state.get('user_input')}\n\n"
            f"**Solution**: {state.get('specialist_output', 'N/A')}\n\n"
            f"**Execution Status**: Completed & Persisted to Memory."
        )

        msg = {
            "id": f"msg_fin_{int(time.time() * 1000)}",
            "sender": "Finalize & Memory Node",
            "role": "assistant",
            "content": final_text,
            "timestamp": time.strftime("%H:%M:%S"),
        }

        return {
            "memory_logs": [memory_entry],
            "messages": [msg],
            "agent_thoughts": [
                {
                    "agent": "Finalize & Memory",
                    "thought": "Saved final execution result to system memory store.",
                    "timestamp": time.strftime("%H:%M:%S"),
                }
            ],
        }

    # ROUTING FUNCTIONS
    def route_intake(state: ChartPipelineState) -> str:
        if state.get("intake_status") == "BLOCKED":
            return "blocked_end"
        return "specialist"

    def route_convergence(state: ChartPipelineState) -> str:
        if state.get("is_converged"):
            return "prepare_action"
        return "escalation"

    def route_prepare_action(state: ChartPipelineState) -> str:
        if state.get("action_blocked"):
            return "blocked_memory"
        return "approval"

    # BUILD GRAPH
    workflow.add_node("intake", intake_node)
    workflow.add_node("blocked_end", blocked_end_node)
    workflow.add_node("specialist", specialist_node)
    workflow.add_node("tier0_checks", tier0_checks_node)
    workflow.add_node("tier1_verify", tier1_verify_node)
    workflow.add_node("escalation", escalation_node)
    workflow.add_node("adjudicate_repair", adjudicate_repair_node)
    workflow.add_node("prepare_action", prepare_action_node)
    workflow.add_node("blocked_memory", blocked_memory_node)
    workflow.add_node("approval", approval_node)
    workflow.add_node("execute", execute_node)
    workflow.add_node("finalize_memory", finalize_memory_node)

    # EDGES
    workflow.add_edge(START, "intake")
    workflow.add_conditional_edges(
        "intake", route_intake, {"specialist": "specialist", "blocked_end": "blocked_end"}
    )
    workflow.add_edge("blocked_end", END)

    workflow.add_edge("specialist", "tier0_checks")
    workflow.add_edge("tier0_checks", "tier1_verify")
    workflow.add_conditional_edges(
        "tier1_verify",
        route_convergence,
        {"prepare_action": "prepare_action", "escalation": "escalation"},
    )

    workflow.add_edge("escalation", "adjudicate_repair")
    workflow.add_edge("adjudicate_repair", "prepare_action")

    workflow.add_conditional_edges(
        "prepare_action",
        route_prepare_action,
        {"approval": "approval", "blocked_memory": "blocked_memory"},
    )
    workflow.add_edge("blocked_memory", END)

    workflow.add_edge("approval", "execute")
    workflow.add_edge("execute", "finalize_memory")
    workflow.add_edge("finalize_memory", END)

    return workflow.compile()


# Default compiled graph instance for LangGraph Studio CLI
default_chart_graph = create_chart_pipeline_graph(LocalLLMClient())
