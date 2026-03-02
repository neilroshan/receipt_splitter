# Receipt Splitter

A LangGraph-powered AI agent that analyzes receipt images and calculates how to split the bill between multiple people.

## How It Works

The agent runs as a stateful graph with the following steps:

1. **Load** - Extracts the image from the incoming message
2. **Analyze** - Uses GPT-4o-mini vision to extract all line items, prices, tax, and subtotal from the receipt
3. **Ask Split** - Interrupts to ask how many people are splitting and their names
4. **Get Split Instructions** - Interrupts to ask how to divide specific items (supports natural language like "Alice pays for item 1", "split items 2-3 equally", etc.)
5. **Calculate Itemization** - Uses an LLM to parse the split instructions and compute each person's share, then distributes tax proportionally

## Setup

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)

### Install dependencies

```bash
uv sync
```

### Environment variables

Copy the example env file and fill in your keys:

```bash
cp .env_example .env
```

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (used for GPT-4o-mini vision and itemization) |
| `LANGSMITH_API_KEY` | LangSmith API key (for tracing) |

### Run the agent

Start the LangGraph development server:

```bash
uv run langgraph dev
```

Then open the LangGraph Studio UI and send a message with a receipt image attached.

## Project Structure

```
receipt_splitter/
├── receipt_agent/
│   ├── agent.py          # Graph definition and compilation
│   └── utils/
│       ├── nodes.py      # Node implementations
│       └── state.py      # State schema and Pydantic models
├── main.py
├── langgraph.json        # LangGraph server config
└── pyproject.toml
```
