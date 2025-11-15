from __future__ import annotations

import asyncio
import io
from pathlib import Path

import pandas as pd
from a2a.server.apps.jsonrpc.fastapi_app import A2AFastAPIApplication
from a2a.server.request_handlers.request_handler import RequestHandler
from a2a.types import (
    AgentCard,
    InvalidParamsError,
    Message,
    MessageSendParams,
    Task,
    TaskIdParams,
    TaskPushNotificationConfig,
    TaskQueryParams,
)
from a2a.utils.errors import ServerError
from fastapi import FastAPI

from src.a2a_utils import (
    create_agent_card,
    get_data_part,
)
from src.agents.autogen_agent import run_autogen_visualizer
from src.agents.crewai_agent import run_crewai_analysis
from src.agents.langchain_agent import run_langchain_reader


def _unsupported(method_name: str) -> ServerError:
    return ServerError(error=InvalidParamsError(message=f"Method '{method_name}' not supported."))


class SimpleRequestHandler(RequestHandler):
    """Base handler that raises unsupported errors for optional methods."""

    async def on_get_task(self, params: TaskQueryParams, context=None) -> Task | None:
        raise _unsupported('tasks/get')

    async def on_cancel_task(self, params: TaskIdParams, context=None) -> Task | None:
        raise _unsupported('tasks/cancel')

    async def on_message_send_stream(self, params: MessageSendParams, context=None):
        raise _unsupported('message/stream')
        yield

    async def on_set_task_push_notification_config(
        self, params: TaskPushNotificationConfig, context=None
    ) -> TaskPushNotificationConfig:
        raise _unsupported('tasks/pushNotificationConfig/set')

    async def on_get_task_push_notification_config(self, params, context=None):  # type: ignore[override]
        raise _unsupported('tasks/pushNotificationConfig/get')

    async def on_resubscribe_to_task(self, params: TaskIdParams, context=None):
        raise _unsupported('tasks/resubscribe')
        yield

    async def on_list_task_push_notification_config(self, params, context=None):  # type: ignore[override]
        raise _unsupported('tasks/pushNotificationConfig/list')

    async def on_delete_task_push_notification_config(self, params, context=None):  # type: ignore[override]
        raise _unsupported('tasks/pushNotificationConfig/delete')


class ReaderRequestHandler(SimpleRequestHandler):
    def __init__(self, llm_model: str) -> None:
        self.llm_model = llm_model

    async def on_message_send(self, params: MessageSendParams, context=None) -> Message:  # type: ignore[override]
        payload = get_data_part(params.message) or {}
        dataset_path = payload.get('dataset_path')
        csv_text = payload.get('csv_text')
        records = payload.get('records')
        llm_model = payload.get('model', self.llm_model)

        dataframe: pd.DataFrame | None = None
        if records:
            dataframe = pd.DataFrame(records)
        elif csv_text:
            dataframe = pd.read_csv(io.StringIO(csv_text))
        elif not dataset_path:
            raise ServerError(
                error=InvalidParamsError(
                    message="Reader agent expects 'dataset_path', 'csv_text', or 'records'."
                )
            )

        def run_reader() -> Message:
            result = run_langchain_reader(
                dataset_path,
                dataframe=dataframe,
                llm_model=llm_model,
            )
            return result['message']

        return await asyncio.to_thread(run_reader)


class AnalystRequestHandler(SimpleRequestHandler):
    def __init__(self, llm_model: str) -> None:
        self.llm_model = llm_model

    async def on_message_send(self, params: MessageSendParams, context=None) -> Message:  # type: ignore[override]
        payload = get_data_part(params.message) or {}
        records = payload.get('records')
        if records is not None and not isinstance(records, list):
            raise ServerError(
                error=InvalidParamsError(
                    message="Analyst agent expected 'records' to be a list of dictionaries."
                )
            )
        summary_text = payload.get('summary_text')
        metrics = payload.get('metrics')
        llm_model = payload.get('model', self.llm_model)

        if not records or summary_text is None or metrics is None:
            raise ServerError(
                error=InvalidParamsError(
                    message="Analyst agent expects 'records', 'summary_text', and 'metrics'."
                )
            )

        def run_analyst() -> Message:
            result = run_crewai_analysis(
                sales_records=records,
                reader_summary=summary_text,
                metrics=metrics,
                llm_model=llm_model,
            )
            return result['message']

        return await asyncio.to_thread(run_analyst)


class VisualizerRequestHandler(SimpleRequestHandler):
    def __init__(self, llm_model: str) -> None:
        self.llm_model = llm_model

    async def on_message_send(self, params: MessageSendParams, context=None) -> Message:  # type: ignore[override]
        payload = get_data_part(params.message) or {}
        records = payload.get('records')
        if records is not None and not isinstance(records, list):
            raise ServerError(
                error=InvalidParamsError(
                    message="Visualizer agent expected 'records' to be a list of dictionaries."
                )
            )
        analysis_summary = payload.get('analysis_text')
        output_dir = payload.get('artifacts_dir', 'artifacts/autogen')
        llm_model = payload.get('model', self.llm_model)

        if not records or analysis_summary is None:
            raise ServerError(
                error=InvalidParamsError(
                    message="Visualizer agent expects 'records' and 'analysis_text'."
                )
            )

        def run_visualizer() -> Message:
            result = run_autogen_visualizer(
                sales_records=records,
                analysis_summary=analysis_summary,
                output_dir=Path(output_dir),
                llm_model=llm_model,
            )
            return result['message']

        return await asyncio.to_thread(run_visualizer)


def _build_app(
    *,
    handler: RequestHandler,
    card: AgentCard,
    rpc_path: str,
) -> FastAPI:
    application = A2AFastAPIApplication(agent_card=card, http_handler=handler)
    return application.build(rpc_url=rpc_path)


def create_reader_app(
    *,
    public_url: str,
    rpc_path: str = '/a2a',
    llm_model: str = 'gpt-4o-mini',
) -> FastAPI:
    card = create_agent_card(
        name='LangChain Reader',
        description='Reads sales datasets and produces structured summaries.',
        skill_id='sales_csv_ingest',
        skill_name='Sales Document Ingestion',
        skill_description='Summarizes sales and marketing CSV content for downstream agents.',
        url=public_url + rpc_path,
    )
    handler = ReaderRequestHandler(llm_model=llm_model)
    return _build_app(handler=handler, card=card, rpc_path=rpc_path)


def create_analyst_app(
    *,
    public_url: str,
    rpc_path: str = '/a2a',
    llm_model: str = 'gpt-4o-mini',
) -> FastAPI:
    card = create_agent_card(
        name='CrewAI Analyst',
        description='Transforms reader output into strategic recommendations.',
        skill_id='sales_revops_analysis',
        skill_name='Revenue Analysis',
        skill_description='Produces markdown briefings and structured analytics for leadership decisions.',
        url=public_url + rpc_path,
    )
    handler = AnalystRequestHandler(llm_model=llm_model)
    return _build_app(handler=handler, card=card, rpc_path=rpc_path)


def create_visualizer_app(
    *,
    public_url: str,
    rpc_path: str = '/a2a',
    llm_model: str = 'gpt-4o',
) -> FastAPI:
    card = create_agent_card(
        name='AutoGen Visualizer',
        description='Generates charts and fine analytics from analyst output.',
        skill_id='sales_visual_story',
        skill_name='Visualization & Fine Analytics',
        skill_description='Creates charts and executive-ready visual narratives.',
        url=public_url + rpc_path,
    )
    handler = VisualizerRequestHandler(llm_model=llm_model)
    return _build_app(handler=handler, card=card, rpc_path=rpc_path)
