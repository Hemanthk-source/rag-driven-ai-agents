"""Interactive CLI for the RAG-driven agent.

Usage:
    export ANTHROPIC_API_KEY=your_key
    python main.py            # uses ./docs
    python main.py my_folder  # uses a custom docs folder
"""
import sys

from agent import RAGAgent
from rag import RAGStore


def main():
    docs_dir = sys.argv[1] if len(sys.argv) > 1 else "docs"
    agent = RAGAgent(RAGStore(docs_dir))
    print("RAG Agent ready. Type 'exit' to quit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if q:
            print(f"\nAgent: {agent.ask(q)}\n")


if __name__ == "__main__":
    main()
