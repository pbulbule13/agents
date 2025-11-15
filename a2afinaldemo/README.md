# Sales & Marketing A2A Agents

This project showcases an agent-to-agent workflow that coordinates three distinct agents using the A2A SDK:

- **LangChain Reader** — ingests a sales and marketing dataset and emits structured insights.
- **CrewAI Analyst** — transforms the reader output into strategic recommendations.
- **AutoGen Visualizer** — generates charts and fine analytics, leveraging tool calls to produce visual artifacts.

Outputs from every agent are rendered in a Streamlit UI so you can inspect the entire conversation as well as agent-specific deliverables.

## Prerequisites

- Python 3.13 (managed automatically by [`uv`](https://docs.astral.sh/uv/))
- An OpenAI API key exported as `OPENAI_API_KEY`

```powershell
# Windows PowerShell example
env:OPENAI_API_KEY = "sk-..."
```

## Installation

```powershell
uv venv .venv
uv pip install -r pyproject.toml
```

> The project already includes a `pyproject.toml` with all required dependencies. Once installed, activate the environment if your shell does not automatically pick it up.

```powershell
.venv\Scripts\Activate.ps1
```

## Running the agents (multi-port setup)

Each agent now exposes an A2A-compliant JSON-RPC endpoint on its own port. Launch them in separate terminals (or background processes):

```powershell
python -m scripts.start_agent --agent reader --port 8001
python -m scripts.start_agent --agent analyst --port 8002
python -m scripts.start_agent --agent visualizer --port 8003
```

Alternatively, you can start all three at once (they share the current console; press Ctrl+C to stop them together):

```powershell
python -m scripts.start_all_agents
```

Each command exposes the agent card at `/.well-known/agent-card.json` and the JSON-RPC endpoint at `/a2a`. Override `--public-host` if you need to advertise a different hostname in the agent card (e.g., when running inside a container).

## Launching the Streamlit UI

With the agents running:

```powershell
streamlit run src/ui/app.py
```

In the **Settings** section of the UI you can adjust both the LLM models and the agent endpoints (defaults match the ports above). The workflow now orchestrates exclusively through the networked A2A interfaces:

1. LangChain Reader receives the dataset payload and emits structured metrics.
2. CrewAI Analyst consumes the reader output via the reader’s A2A response.
3. AutoGen Visualizer pulls the analyst summary and generates chart artifacts, returning their paths.

The **Flow Overview** tab lists the endpoints in use, the status log, and the raw AutoGen trace for debugging.

## Command-line pipeline (optional)

If the three agent servers are up, you can still trigger the pipeline from the terminal:

```powershell
python -m scripts.run_pipeline
```

This prints each agent’s summary to stdout and lists the generated Plotly JSON chart files under `artifacts/autogen/`.

## Orchestration Flow

```
Coordinator → Reader (http://localhost:8001/a2a)
Reader ➜ Analyst (http://localhost:8002/a2a)
Analyst ➜ Visualizer (http://localhost:8003/a2a)
```

Every hop exchanges A2A `Message` objects with the earlier agent’s structured payload serialized in `DataPart`s, preserving pure function-based logic.

## Sample Data

`data/sales_marketing.csv` contains a synthetic January 2025 pipeline covering sales, channel, and marketing spend metrics. Feel free to replace it with your own dataset as long as the headers align with the demo (Date, Region, Product, Channel, Sales, Marketing_Spend, Qualified_Leads, New_Customers).

## Notes

- AutoGen requires the environment variable `OPENAI_API_KEY`. Without it, the visualizer agent will raise an error.
- Plotly figures are persisted as JSON files under `artifacts/autogen`. Re-running the workflow will overwrite previous artifacts.
- All project code follows a function-based approach with no custom classes, using library primitives for agent construction.
- Disable CrewAI tracing if your environment cannot reach `app.crewai.com`:
  ```powershell
  env:CREWAI_DISABLE_TRACING = "true"
  ```
- Keep your actual API key only in `.env` (do not expose it in committed files).

