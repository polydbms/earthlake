import os
import sys

from REMSA.agent_orchestrator import FMSAgent

def run_cli():
    agent = FMSAgent()

    print("Welcome to RESMA: Remote Sensing Foundation Model Selection Agent")
    print("Type your model requirement below. Type 'exit' to quit.")

    while True:
        user_input = input("\n[User] >> ")
        if user_input.lower() in ["exit", "quit"]:
            break
        try:
            response, response_db = agent.run(user_input)
            print(f"\n{response}")
        except Exception as e:
            print(f"[Error] {e}")

def run_envar():
    query = os.getenv("QUERY")
    if not query:
        print("Missing QUERY environment variable.")
        sys.exit(1)
    agent = FMSAgent()
    response, response_db = agent.run(query)
    print(response)

if __name__ == "__main__":
    # interactive when no envvar is set
    if os.getenv("QUERY"):
        print("Running REMSA with query '{}'".format(os.getenv("QUERY")))
        run_envar()
    else:
        run_cli()
