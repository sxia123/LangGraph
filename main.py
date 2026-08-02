import sys
import time

from src.agents import (
    create_chart_pipeline_graph,
    create_claims_triage_graph,
    create_code_review_team_graph,
    create_master_pipeline_graph,
    create_multi_agent_supervisor_graph,
    create_solution_review_team_graph,
)
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


def run_chart_pipeline_demo(prompt: str):
    print(f"\n🚀 Launching Flowchart Architecture Pipeline for Prompt:\n'{prompt}'\n")
    graph = create_chart_pipeline_graph(llm_client)

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
        "current_step": "intake",
        "agent_thoughts": [],
    }

    step = 1
    for chunk in graph.stream(initial_input):
        for node_name, node_update in chunk.items():
            print("\n------------------------------------------------------")
            print(f"📍 Step {step}: Node [{node_name.upper()}] Complete")
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
    print("✅ Flowchart Architecture Pipeline Complete!")
    print("======================================================\n")


def run_code_review_demo(prompt: str):
    print(f"\n🚀 Launching Code Review Team Pipeline for Task:\n'{prompt}'\n")
    graph = create_code_review_team_graph(llm_client)

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
        "task": prompt,
        "code": "",
        "review": "",
        "approved": False,
        "revision_count": 0,
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
                print(
                    f"💬 Output ({last_msg.get('sender')}):\n{last_msg.get('content', '').strip()}"
                )

            step += 1

    print("\n======================================================")
    print("✅ Code Review Team Pipeline Complete!")
    print("======================================================\n")


def run_solution_review_demo(prompt: str):
    print(f"\n🚀 Launching Solution Review Team Pipeline for Task:\n'{prompt}'\n")
    graph = create_solution_review_team_graph(llm_client)

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
        "task": prompt,
        "solution": "",
        "review": "",
        "approved": False,
        "revision_count": 0,
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
                print(
                    f"💬 Output ({last_msg.get('sender')}):\n{last_msg.get('content', '').strip()}"
                )

            step += 1

    print("\n======================================================")
    print("✅ Solution Review Team Pipeline Complete!")
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
        print(f"   (Endpoint Target: {llm_client.config.base_url})")
        print("   (Ensure oMLX server is running on port 8000. Falling back to simulation if needed)\n")

    print("\nSelect Agent Flow to Execute:")
    print("1) Multi-Agent Supervisor Team (Researcher + Coder + Critic + Writer)")
    print("2) Claims & Severity Triage Pipeline (3-Step Assessment)")
    print("3) Flowchart Architecture Pipeline (Intake -> Specialist -> Tier0/1 -> Escalation -> Execute)")
    print("4) Code Review Team Pipeline (Developer + Code Auditor)")
    print("5) Solution Review Team Pipeline (Specialist + Quality Auditor)")
    print("6) Master Integrated Pipeline (Triage -> Supervisor -> Quality Auditor)")

    try:
        choice = input("\nEnter selection [1-6] (default 6): ").strip() or "6"
        prompt = (
            input("Enter task prompt: ").strip()
            or "Build a Python function to sanitize user input and test it."
        )

        if choice == "1":
            run_supervisor_demo(prompt)
        elif choice == "2":
            run_claims_triage_demo(prompt)
        elif choice == "3":
            run_chart_pipeline_demo(prompt)
        elif choice == "4":
            run_code_review_demo(prompt)
        elif choice == "5":
            run_solution_review_demo(prompt)
        else:
            run_master_pipeline_demo(prompt)
    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()
