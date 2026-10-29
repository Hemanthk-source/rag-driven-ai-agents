# RAG-Driven AI Agents

A minimal agent that decides *when* to retrieve knowledge from your documents
(RAG) and *when* to use other tools, then answers with citations.

## Architecture
User question -> Claude (agent loop) -> tool calls:
  - `search_knowledge_base`: embeds the query, finds top-k chunks (cosine similarity)
  - `calculator`: safe arithmetic
-> grounded answer with [source] citations

## Setup
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=your_key

## Run
    python main.py

Drop your own `.txt` / `.md` files into `docs/` and ask questions.
Try: "How many leave days can I carry over?" or
"What do 3 years of the Pro plan cost?"

## Files
- `rag.py`   – chunking, embeddings, retrieval
- `tools.py` – tool schemas + calculator
- `agent.py` – agent loop (tool use)
- `main.py`  – CLI
