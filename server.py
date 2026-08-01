import os
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agents.claims_triage_team import create_claims_triage_graph
from src.agents.master_pipeline import create_master_pipeline_graph
from src.agents.multi_agent_supervisor import create_multi_agent_supervisor_graph
from src.core.local_llm import LocalLLMClient

app = FastAPI(title="LangGraph Web API Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm_client = LocalLLMClient()


class ChatRequest(BaseModel):
    prompt: str
    pipeline: Optional[str] = "master"  # "master", "supervisor", "claims_triage"


@app.get("/api/status")
def get_status():
    """Returns local LLM provider configuration and connection status."""
    conn = llm_client.ping()
    return {
        "provider": llm_client.config.provider,
        "base_url": llm_client.config.base_url,
        "model_name": llm_client.config.model_name,
        "connection": conn,
    }


@app.post("/api/chat")
def handle_chat(req: ChatRequest):
    """Executes selected LangGraph pipeline and returns step outputs, thoughts, and final response."""
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    pipeline_choice = (req.pipeline or "master").lower()
    timestamp = time.strftime("%H:%M:%S")
    user_msg_id = f"user_{int(time.time() * 1000)}"

    steps_data = []

    try:
        if pipeline_choice == "supervisor":
            graph = create_multi_agent_supervisor_graph(llm_client)
            initial_input = {
                "messages": [
                    {
                        "id": user_msg_id,
                        "sender": "User",
                        "role": "user",
                        "content": prompt,
                        "timestamp": timestamp,
                    }
                ],
                "current_task": prompt,
                "next_agent": "supervisor",
                "research_output": "",
                "coder_output": "",
                "critic_feedback": "",
                "final_response": "",
                "agent_thoughts": [],
            }
        elif pipeline_choice == "claims_triage":
            graph = create_claims_triage_graph(llm_client)
            initial_input = {
                "messages": [
                    {
                        "id": user_msg_id,
                        "sender": "User",
                        "role": "user",
                        "content": prompt,
                        "timestamp": timestamp,
                    }
                ],
                "claim_input": prompt,
                "current_step": "step_1_classification",
                "classification_details": None,
                "severity_assessment": None,
                "action_plan": None,
                "final_response": "",
                "agent_thoughts": [],
            }
        else:  # Default to master pipeline
            graph = create_master_pipeline_graph(llm_client)
            initial_input = {
                "messages": [
                    {
                        "id": user_msg_id,
                        "sender": "User",
                        "role": "user",
                        "content": prompt,
                        "timestamp": timestamp,
                    }
                ],
                "user_input": prompt,
                "current_step": "pipeline_start",
                "triage_details": None,
                "supervisor_details": None,
                "review_details": None,
                "final_response": "",
                "agent_thoughts": [],
            }

        step_idx = 1
        for chunk in graph.stream(initial_input):
            for node_name, node_update in chunk.items():
                thoughts = node_update.get("agent_thoughts") or []
                messages = node_update.get("messages") or []
                final_resp = node_update.get("final_response") or ""

                steps_data.append(
                    {
                        "step": step_idx,
                        "node": node_name,
                        "thoughts": thoughts,
                        "messages": messages,
                        "final_response": final_resp,
                    }
                )
                step_idx += 1

        # Extract final answer
        final_answer = ""
        for s in reversed(steps_data):
            if s.get("final_response"):
                final_answer = s["final_response"]
                break
            if s.get("messages"):
                last_m = s["messages"][-1]
                if isinstance(last_m, dict) and last_m.get("content"):
                    final_answer = last_m["content"]
                    break

        return {
            "status": "success",
            "pipeline": pipeline_choice,
            "prompt": prompt,
            "steps": steps_data,
            "final_response": final_answer or "Pipeline execution completed.",
        }

    except Exception as err:
        raise HTTPException(
            status_code=500, detail=f"Pipeline execution error: {str(err)}"
        ) from err


# Serve static files from public directory
public_dir = os.path.join(os.path.dirname(__file__), "public")
if os.path.exists(public_dir):
    app.mount("/", StaticFiles(directory=public_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
