import sys
import time

from src.agents.claims_triage_team import create_claims_triage_graph
from src.agents.master_pipeline import create_master_pipeline_graph
from src.agents.multi_agent_supervisor import create_multi_agent_supervisor_graph
from src.core.local_llm import LocalLLMClient

llm_client = LocalLLMClient()


def print_header():
    print("\n======================================================")
    print("     LangGraph Python CLI - Multi-Agent Framework")
    print("======================================================")
    print(f"Configured Provider: [{llm_client.config.provider.upper()}]")
    print(f"Endpoint Base URL : {llm_client.config.base_url}")
    print(f"Model Name        : {llm_client.config.model_name}")
    print("------------------------------------------------------\n")


def run_supervisor_demo(prompt: str):
    print(f"\n🚀 Launching Multi-Agent Supervisor Team for Task:\n'{prompt}'\n")
    graph = create_multi_agent_supervisor_graph(llm_client)

    initial_input = {
        "messages": [
            {
                "id": f"user_{int(time.time() * 1000)}",
                "sender": "User",
                "role": "user",
                "content": prompt,
                "timestamp": time.strftime("%H:%M:%S"),
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

    step = 1
    for chunk in graph.stream(initial_input):
        for node_name, node_update in chunk.items():
            print("\n------------------------------------------------------")
            print(f"📍 Step {step}: Node [{node_name.upper()}] Executed")
            print("------------------------------------------------------")

            thoughts = node_update.get("agent_thoughts", [])
            if thoughts:
                last_thought = thoughts[-1]
                print(f"🧠 Thought [{last_thought.get('agent')}]: {last_thought.get('thought')}")

            messages = node_update.get("messages", [])
            if messages:
                last_msg = messages[-1]
                print(
                    f"💬 Output ({last_msg.get('sender')}):\n{last_msg.get('content', '').strip()}"
                )

            step += 1

    print("\n======================================================")
    print("✅ Multi-Agent Supervisor Execution Complete!")
    print("======================================================\n")


def run_claims_triage_demo(prompt: str):
    print(f"\n🚀 Launching Claims & Severity Triage Pipeline for Prompt:\n'{prompt}'\n")
    graph = create_claims_triage_graph(llm_client)

    initial_input = {
        "messages": [
            {
                "id": f"user_{int(time.time() * 1000)}",
                "sender": "User",
                "role": "user",
                "content": prompt,
                "timestamp": time.strftime("%H:%M:%S"),
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

    step = 1
    for chunk in graph.stream(initial_input):
        for node_name, node_update in chunk.items():
            print("\n------------------------------------------------------")
            print(f"📍 Step {step}: Node [{node_name.upper()}] Complete")
            print("------------------------------------------------------")

            messages = node_update.get("messages", [])
            if messages:
                last_msg = messages[-1]
                print(f"{last_msg.get('content', '').strip()}")

            step += 1

    print("\n======================================================")
    print("✅ Claims Pipeline Complete!")
    print("======================================================\n")


def run_master_pipeline_demo(prompt: str):
    print(f"\n🚀 Launching Master Integrated Pipeline for Prompt:\n'{prompt}'\n")
    graph = create_master_pipeline_graph(llm_client)

    initial_input = {
        "messages": [
            {
                "id": f"user_{int(time.time() * 1000)}",
                "sender": "User",
                "role": "user",
                "content": prompt,
                "timestamp": time.strftime("%H:%M:%S"),
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

    step = 1
    for chunk in graph.stream(initial_input):
        for node_name, node_update in chunk.items():
            print("\n------------------------------------------------------")
            print(f"📍 Stage {step}: Node [{node_name.upper()}] Complete")
            print("------------------------------------------------------")

            messages = node_update.get("messages", [])
            if messages:
                last_msg = messages[-1]
                print(f"💬 Output ({last_msg.get('sender')}):\n{last_msg.get('content', '').strip()}")

            step += 1

    print("\n======================================================")
    print("✅ Master Integrated Pipeline Complete!")
    print("======================================================\n")


def main():
    print_header()
    print("Testing connection to local AI endpoint...")
    test_res = llm_client.ping()
    if test_res.get("ok"):
        print(f"✅ {test_res.get('message')}")
    else:
        print(f"⚠️ {test_res.get('message')}")
        print("   (Falling back to local simulation mode if requests fail)\n")

    print("\nSelect Agent Flow to Execute:")
    print("1) Multi-Agent Supervisor Team (Researcher + Coder + Critic + Writer)")
    print("2) Claims & Severity Triage Pipeline (3-Step Assessment)")
    print("3) Master Integrated Pipeline (Triage -> Supervisor -> Code Auditor)")

    try:
        choice = input("\nEnter selection [1-3] (default 3): ").strip() or "3"
        prompt = (
            input("Enter task prompt: ").strip()
            or "Build a Python function to sanitize user input and test it."
        )

        if choice == "1":
            run_supervisor_demo(prompt)
        elif choice == "2":
            run_claims_triage_demo(prompt)
        else:
            run_master_pipeline_demo(prompt)
    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()
