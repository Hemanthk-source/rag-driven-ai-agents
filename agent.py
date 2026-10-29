"""RAG-driven agent: Claude decides when to retrieve, calculate, or answer."""
import os

import anthropic

from rag import RAGStore
from tools import TOOL_SCHEMAS, calculator

SYSTEM_PROMPT = (
    "You are a helpful research assistant. For questions about the user's "
    "documents, call search_knowledge_base first (you may search several "
    "times with different queries). Ground your answers in the retrieved "
    "passages and cite the source filename in brackets, e.g. [handbook.md]. "
    "If the knowledge base doesn't contain the answer, say so plainly. "
    "Use the calculator for any arithmetic."
)


class RAGAgent:
    def __init__(self, store: RAGStore, model=None, max_steps=6):
        self.client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY
        self.model = model or os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
        self.store = store
        self.max_steps = max_steps
        self.history = []

    def _run_tool(self, name, args):
        if name == "search_knowledge_base":
            hits = self.store.search(args["query"], args.get("top_k", 3))
            return "\n\n".join(
                f"[{h['source']}] (score {h['score']:.2f})\n{h['text']}" for h in hits
            )
        if name == "calculator":
            return calculator(args["expression"])
        return f"Unknown tool: {name}"

    def ask(self, question: str, verbose=True) -> str:
        self.history.append({"role": "user", "content": question})

        for _ in range(self.max_steps):
            resp = self.client.messages.create(
                model=self.model, max_tokens=1024, system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS, messages=self.history,
            )
            self.history.append({"role": "assistant", "content": resp.content})

            if resp.stop_reason != "tool_use":
                return "".join(b.text for b in resp.content if b.type == "text")

            results = []
            for block in resp.content:
                if block.type == "tool_use":
                    if verbose:
                        print(f"  🔧 {block.name}({block.input})")
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": self._run_tool(block.name, block.input),
                    })
            self.history.append({"role": "user", "content": results})

        return "Stopped: reached the maximum number of reasoning steps."
