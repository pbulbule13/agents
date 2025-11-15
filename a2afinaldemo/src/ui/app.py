from __future__ import annotations

from pathlib import Path
from typing import Any

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import plotly.io as pio
import streamlit as st
from dotenv import load_dotenv

from src.orchestrator import orchestrate_sales_insights


def _persist_upload(upload) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / upload.name
    file_path.write_bytes(upload.getbuffer())
    return file_path


def _render_plot(artifact: dict[str, Any]) -> None:
    path = artifact.get("figure_path")
    if not path:
        st.info("Tool output did not provide a figure path.")
        return
    figure_file = Path(path)
    if not figure_file.exists():
        st.warning(f"Figure not found at {figure_file}.")
        st.json(artifact)
        return
    try:
        figure_json = figure_file.read_text(encoding="utf-8")
        figure = pio.from_json(figure_json)
        st.plotly_chart(figure, use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unable to render chart: {exc}")
    st.caption(f"Source: {figure_file}")


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Sales & Marketing A2A Agents", layout="wide")
    st.title("Sales & Marketing Agent-to-Agent Orchestration")
    st.write(
        "This demo coordinates LangChain, CrewAI, and AutoGen agents via the A2A SDK. "
        "Provide a sales CSV to see each agent's output."
    )

    default_dataset = Path("data/sales_marketing.csv")
    dataset_choice = st.radio(
        "Choose a dataset",
        ("Use bundled sample", "Upload CSV"),
        horizontal=True,
    )

    dataset_path: Path | None = None
    active_dataframe = None

    if dataset_choice == "Use bundled sample":
        if not default_dataset.exists():
            st.error("Sample dataset is missing. Please upload your own CSV.")
        else:
            dataset_path = default_dataset
            active_dataframe = pd.read_csv(default_dataset)
            st.dataframe(active_dataframe)
    else:
        uploaded = st.file_uploader("Upload a sales & marketing CSV", type="csv")
        if uploaded is not None:
            dataset_path = _persist_upload(uploaded)
            st.success(f"Uploaded dataset stored at {dataset_path}")
            active_dataframe = pd.read_csv(dataset_path)
            st.dataframe(active_dataframe)

    override_col1, override_col2, override_col3 = st.columns(3)
    with override_col1:
        langchain_model = st.text_input("LangChain model", value="gpt-4o-mini")
    with override_col2:
        crewai_model = st.text_input("CrewAI model", value="gpt-4o-mini")
    with override_col3:
        autogen_model = st.text_input("AutoGen model", value="gpt-4o")

    endpoint_col1, endpoint_col2, endpoint_col3 = st.columns(3)
    with endpoint_col1:
        reader_endpoint = st.text_input(
            "Reader endpoint",
            value="http://localhost:8001/a2a",
        )
    with endpoint_col2:
        analyst_endpoint = st.text_input(
            "Analyst endpoint",
            value="http://localhost:8002/a2a",
        )
    with endpoint_col3:
        visualizer_endpoint = st.text_input(
            "Visualizer endpoint",
            value="http://localhost:8003/a2a",
        )

    run_button = st.button("Run Agents", type="primary", disabled=dataset_path is None)

    if not run_button:
        st.stop()

    if dataset_path is None:
        st.error("Please select or upload a dataset before running the agents.")
        st.stop()

    status_messages: list[str] = []

    with st.status("Running agent-to-agent workflow...", expanded=True) as status:
        status.write("Launching LangChain Reader…")

        def progress(step: str, payload: dict[str, Any]) -> None:
            if step == "reader":
                status.write("✅ LangChain Reader completed")
            elif step == "analyst":
                status.write("✅ CrewAI Analyst completed")
            elif step == "visualizer":
                status.write("✅ AutoGen Visualizer completed")

        try:
            results = orchestrate_sales_insights(
                dataset_path,
                model_overrides={
                    "langchain": langchain_model,
                    "crewai": crewai_model,
                    "autogen": autogen_model,
                },
                agent_endpoints={
                    "reader": reader_endpoint,
                    "analyst": analyst_endpoint,
                    "visualizer": visualizer_endpoint,
                },
                progress_callback=progress,
            )
        except Exception as exc:  # noqa: BLE001
            status.update(label="Workflow failed", state="error")
            st.error(f"Failed to run agents: {exc}")
            st.stop()
        status.update(label="Agents completed", state="complete")

    st.success("Agent collaboration complete.")

    flow_tab, summary_tab, analyst_tab, visual_tab, convo_tab = st.tabs(
        [
            "Flow Overview",
            "LangChain Reader",
            "CrewAI Analyst",
            "AutoGen Visualizer",
            "Conversation",
        ]
    )

    with flow_tab:
        st.markdown("### A2A Coordination Timeline")
        st.markdown("LangChain Reader ➜ CrewAI Analyst ➜ AutoGen Visualizer")
        st.markdown("#### Agent Endpoints")
        st.json(
            {
                "reader": reader_endpoint,
                "analyst": analyst_endpoint,
                "visualizer": visualizer_endpoint,
            }
        )
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Step 1", "LangChain Reader")
        with col_b:
            st.metric("Step 2", "CrewAI Analyst")
        with col_c:
            st.metric("Step 3", "AutoGen Visualizer")

        st.markdown("#### Reader Snapshot")
        st.write(results["reader"]["summary"])

        st.markdown("#### Analyst Snapshot")
        st.write(results["analyst"]["analysis"])

        st.markdown("#### Visualizer Summary")
        st.write(results["visualizer"]["insights"])

        st.markdown("#### AutoGen Trace")
        with st.expander("View raw AutoGen messages"):
            for idx, raw in enumerate(results["visualizer"].get("raw_messages", []), start=1):
                st.code(f"[{idx}] {raw}")

    with summary_tab:
        st.subheader("Reader Summary")
        st.write(results["reader"]["summary"])
        st.subheader("Metrics Snapshot")
        st.json(results["reader"]["metrics"])
        st.subheader("Agent Card")
        st.json(results["cards"]["reader"].model_dump())

    with analyst_tab:
        st.subheader("CrewAI Narrative")
        st.write(results["analyst"]["analysis"])
        if results["analyst"]["structured"]:
            st.subheader("Structured Analytics JSON")
            st.json(results["analyst"]["structured"])
        st.subheader("Agent Card")
        st.json(results["cards"]["analyst"].model_dump())

    with visual_tab:
        st.subheader("Visualizer Insights")
        st.write(results["visualizer"]["insights"])
        st.subheader("Generated Charts")
        if results["visualizer"]["tool_outputs"]:
            for artifact in results["visualizer"]["tool_outputs"]:
                _render_plot(artifact)
                st.json(artifact)
        else:
            st.info("No chart artifacts detected from the AutoGen agent.")
        st.subheader("AutoGen Messages")
        for idx, raw in enumerate(results["visualizer"].get("raw_messages", []), start=1):
            st.code(f"[{idx}] {raw}")
        st.subheader("Agent Card")
        st.json(results["cards"]["visualizer"].model_dump())

    with convo_tab:
        st.subheader("Agent Message Log")
        for entry in results["conversation_log"]:
            st.markdown(f"### {entry['speaker']}")
            st.write(entry["text"])
            if entry["data"]:
                with st.expander("Structured payload"):
                    st.json(entry["data"])


if __name__ == "__main__":
    main()
