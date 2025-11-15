from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.a2a_utils import create_agent_message


def _load_dataframe(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Sales dataset not found: {path}")
    return pd.read_csv(path)


def _build_prompt() -> ChatPromptTemplate:
    template = (
        "You are a revenue operations ingestion agent. "
        "Review the provided sales and marketing records and craft a concise "
        "overview. Highlight demand trends, channel efficiency, and any "
        "anomalies worth deeper analysis."
        "\n\nDataset preview:\n{preview}\n\n"
        "Provide:\n"
        "1. Three bullet insights on revenue and pipeline momentum.\n"
        "2. Two risks or anomalies worth escalation.\n"
        "3. A JSON block named metrics summarizing totals for sales, marketing_spend, "
        "lead_to_customer_rate, and regions_ranked."
    )
    return ChatPromptTemplate.from_messages([("system", template)])


def _compute_metrics(df: pd.DataFrame) -> dict[str, Any]:
    total_sales = float(df["Sales"].sum())
    total_spend = float(df["Marketing_Spend"].sum())
    leads = float(df["Qualified_Leads"].sum())
    customers = float(df["New_Customers"].sum())
    lead_to_customer_rate = round(customers / leads, 4) if leads else 0.0
    avg_order = round(total_sales / customers, 2) if customers else 0.0
    spend_efficiency = round(total_sales / total_spend, 2) if total_spend else math.inf

    return {
        "totals": {
            "sales ": total_sales,
            "marketing_spend": total_spend,
            "qualified_leads": leads,
            "new_customers": customers,
        },
        "efficiency": {
            "lead_to_customer_rate": lead_to_customer_rate,
            "revenue_per_customer": avg_order,
            "revenue_per_dollar": spend_efficiency,
        },
        "regions_ranked": (
            df.groupby("Region")["Sales"].sum().sort_values(ascending=False).to_dict()
        ),
        "products_ranked": (
            df.groupby("Product")["Sales"].sum().sort_values(ascending=False).to_dict()
        ),
        "channels_ranked": (
            df.groupby("Channel")["Sales"].sum().sort_values(ascending=False).to_dict()
        ),
    }


def _format_preview(df: pd.DataFrame, rows: int = 5) -> str:
    preview_df = df.head(rows)
    return preview_df.to_csv(index=False)


def run_langchain_reader(
    csv_path: str | Path | None,
    *,
    dataframe: pd.DataFrame | None = None,
    llm_model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Load the dataset and produce a structured LangChain summary."""
    if dataframe is None:
        if csv_path is None:
            raise ValueError("Either csv_path or dataframe must be provided to run_langchain_reader.")
        dataframe = _load_dataframe(csv_path)
    metrics = _compute_metrics(dataframe)
    preview = _format_preview(dataframe)

    if csv_path is not None:
        loader = CSVLoader(str(csv_path))
        documents = loader.load()
        joined_rows = "\n".join(doc.page_content for doc in documents)
    else:
        joined_rows = dataframe.to_csv(index=False)

    llm = ChatOpenAI(model=llm_model, temperature=temperature)
    prompt = _build_prompt()
    response = llm.invoke(prompt.format_messages(preview=preview + "\n" + joined_rows))
    summary_text = response.content.strip()

    message = create_agent_message(
        text=summary_text,
        data={
            "schema": list(dataframe.columns),
            "records": dataframe.to_dict(orient="records"),
            "metrics": metrics,
        },
    )

    return {
        "message": message,
        "summary_text": summary_text,
        "dataframe": dataframe,
        "records": dataframe.to_dict(orient="records"),
        "metrics": metrics,
        "preview": preview,
    }
