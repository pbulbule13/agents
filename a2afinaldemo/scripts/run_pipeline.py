from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from src.orchestrator import orchestrate_sales_insights


def main() -> None:
    load_dotenv()
    dataset = Path("data/sales_marketing.csv")
    results = orchestrate_sales_insights(dataset)

    print("=== LangChain Reader Summary ===")
    print(results["reader"]["summary"])

    print("\n=== CrewAI Analyst Highlights ===")
    print(results["analyst"]["analysis"])

    print("\n=== AutoGen Visualizer Insights ===")
    print(results["visualizer"]["insights"])

    print("\nGenerated chart artifacts:")
    if results["visualizer"]["tool_outputs"]:
        for artifact in results["visualizer"]["tool_outputs"]:
            print(f" - {artifact.get('figure_path')}")
    else:
        print("No chart artifacts reported.")


if __name__ == "__main__":
    main()
