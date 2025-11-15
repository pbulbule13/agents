from __future__ import annotations

import json
from typing import Any

from crewai import Agent, Crew, Task
from crewai.process import Process
from langchain_openai import ChatOpenAI

from src.a2a_utils import create_agent_message


def run_crewai_analysis(
    *,
    sales_records: list[dict[str, Any]],
    reader_summary: str,
    metrics: dict[str, Any],
    llm_model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Run a CrewAI analyst over the sales data."""
    llm = ChatOpenAI(model=llm_model, temperature=temperature)

    analyst = Agent(
        role="Revenue Intelligence Analyst",
        goal="Transform sales and marketing signals into strategic guidance",
        backstory=(
            "Seasoned RevOps partner focused on pipeline health, marketing ROI, and "
            "regional performance swings. Skilled at converting noisy channel data "
            "into crisp executive briefings."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    context_blob = json.dumps(
        {
            "summary": reader_summary,
            "metrics": metrics,
            "sample_records": sales_records[:8],
        },
        indent=2,
    )

    task = Task(
        description=(
            "Use the provided sales intelligence context to draft a decision-ready "
            "analysis. Context:\n{{context}}\n\n"
            "Break down demand generation momentum, CAC efficiency, conversion "
            "bottlenecks, and channel ROI."
        ),
        expected_output=(
            "Return a markdown report with sections for Momentum, Efficiency, Risks, "
            "and Recommendations. Finish with a ```json``` block named analytics_json "
            "containing keys: kpis, risk_alerts, acceleration_plays, forecast_notes."
        ),
        agent=analyst,
    )

    crew = Crew(
        name="sales_analytics_crew",
        agents=[analyst],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    output = crew.kickoff(inputs={"context": context_blob})

    structured = output.json_dict or {}
    message_text = output.raw.strip() if output.raw else json.dumps(structured, indent=2)

    message = create_agent_message(
        text=message_text,
        data={
            "analytics_json": structured,
            "task_outputs": [task_output.model_dump() for task_output in output.tasks_output],
        },
    )

    return {
        "message": message,
        "analysis_text": message_text,
        "structured": structured,
        "crew_output": output.model_dump(),
    }
