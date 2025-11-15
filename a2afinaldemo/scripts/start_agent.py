from __future__ import annotations

import argparse

import uvicorn
from dotenv import load_dotenv

from src.a2a_servers import (
    create_analyst_app,
    create_reader_app,
    create_visualizer_app,
)


AGENT_BUILDERS = {
    'reader': create_reader_app,
    'analyst': create_analyst_app,
    'visualizer': create_visualizer_app,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Start an A2A agent server.')
    parser.add_argument(
        '--agent',
        choices=AGENT_BUILDERS.keys(),
        required=True,
        help='Agent to launch (reader, analyst, visualizer).',
    )
    parser.add_argument('--host', default='0.0.0.0', help='Bind host for the server.')
    parser.add_argument('--port', type=int, required=True, help='Port to listen on.')
    parser.add_argument(
        '--public-host',
        default='localhost',
        help='Host clients should use when connecting (used in the agent card URL).',
    )
    parser.add_argument(
        '--rpc-path',
        default='/a2a',
        help='Path for the JSON-RPC endpoint (default: /a2a).',
    )
    parser.add_argument(
        '--llm-model',
        default=None,
        help='Optional override for the agent LLM model identifier.',
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    builder = AGENT_BUILDERS[args.agent]

    llm_kwargs = {}
    if args.llm_model:
        llm_kwargs['llm_model'] = args.llm_model

    public_url = f"http://{args.public_host}:{args.port}"
    app = builder(public_url=public_url, rpc_path=args.rpc_path, **llm_kwargs)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
