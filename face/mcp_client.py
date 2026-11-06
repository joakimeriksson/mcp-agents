"""
MCP server configuration for the face agent.

Gives the LLM access to external tools (smart home, web search, etc.)
via MCP (Model Context Protocol) servers.

--- How to add a new MCP server ---

Option A: Config file (mcp_servers.json)

    {
        "servers": [
            {
                "name": "lights",
                "description": "Control smart home lights (on/off, brightness, color)",
                "type": "sse",
                "url": "http://localhost:8000/sse"
            },
            {
                "name": "search",
                "description": "Search the web for information",
                "type": "stdio",
                "command": "uv",
                "args": ["--directory", "../my_mcp", "run", "server.py"]
            }
        ]
    }

The "description" field is shown to the LLM so it can explain its own
capabilities in conversation (e.g. "I can control your lights").

Option B: CLI flag

    python agent.py --mcp-server http://localhost:8000/sse

Server types:
  sse   — connect to an already-running MCP server over HTTP
  stdio — launch an MCP server as a subprocess (stdin/stdout)
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from pydantic_ai.mcp import MCPServerSSE, MCPServerStdio

logger = logging.getLogger("mcp_client")

DEFAULT_CONFIG = "mcp_servers.json"


class ServiceHost:
    """A 'service' MCP server (like candytron_mcp) that, besides tools,
    publishes a persona and lifecycle for its client:

      - resource ``url://get_service_name``  — the service's display name
      - prompt   ``get_service_prompt(lang)`` — persona/system prompt snippet
      - resource ``url://service_init``      — call once before using tools
      - resource ``url://service_exit``      — call on shutdown

    All of these are optional — a server that lacks one simply returns None
    and the agent keeps its defaults. Each call opens a short-lived SSE
    session, so no connection is held between calls.
    """

    def __init__(self, url: str):
        self._url = url

    def _run(self, coro, label: str):
        try:
            return asyncio.run(coro)
        except Exception as e:
            logger.warning(f"service {label} failed ({self._url}): {e}")
            return None

    async def _read_resource(self, uri: str):
        from fastmcp import Client
        from fastmcp.client.transports import SSETransport
        async with Client(transport=SSETransport(self._url)) as client:
            res = await client.read_resource(uri)
            return res[0].text

    async def _get_prompt(self, name: str, lang: str):
        from fastmcp import Client
        from fastmcp.client.transports import SSETransport
        async with Client(transport=SSETransport(self._url)) as client:
            prompts = {p.name: p for p in await client.list_prompts()}
            if name not in prompts:
                return None
            args = {}
            if any(a.name == "lang" for a in prompts[name].arguments or []):
                args["lang"] = lang
            pr = await client.get_prompt(name, args)
            return pr.messages[0].content.text

    def fetch_name(self) -> Optional[str]:
        return self._run(self._read_resource("url://get_service_name"), "name")

    def fetch_prompt(self, lang: str) -> Optional[str]:
        return self._run(self._get_prompt("get_service_prompt", lang), "prompt")

    def fetch_augmentation(self, lang: str) -> Optional[str]:
        """Per-turn state from the service (e.g. candytron's current candy
        positions from the vision system), to inject before the user prompt."""
        return self._run(self._get_prompt("get_service_augmentation", lang),
                         "augmentation")

    def init(self) -> bool:
        return self._run(self._read_resource("url://service_init"),
                         "init") is not None

    def exit(self) -> bool:
        return self._run(self._read_resource("url://service_exit"),
                         "exit") is not None


def load_servers(
    config_path: Optional[str] = None,
    server_urls: Optional[list[str]] = None,
) -> tuple[list, list[str]]:
    """Load MCP servers from a config file and/or CLI URLs.

    Returns:
        (servers, descriptions) — servers are pydantic-ai toolset objects,
        descriptions are human-readable strings for the system prompt.
    """
    servers = []
    descriptions = []

    # --- Config file ---
    path = Path(config_path or DEFAULT_CONFIG)
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        for entry in data.get("servers", []):
            server = _create_server(entry)
            if server:
                servers.append(server)
                desc = entry.get("description", entry.get("name", "unnamed tool"))
                descriptions.append(desc)
        logger.info(f"Loaded {len(servers)} MCP server(s) from {path}")
    elif config_path:
        logger.warning(f"MCP config not found: {path}")

    # --- CLI URLs (SSE) ---
    for url in server_urls or []:
        logger.info(f"MCP server (CLI): SSE -> {url}")
        servers.append(MCPServerSSE(url=url))
        descriptions.append(f"Tools at {url}")

    if servers:
        logger.info(f"Total MCP servers: {len(servers)}")
    return servers, descriptions


def _create_server(entry: dict):
    """Create a single MCP server from a config dict."""
    name = entry.get("name", "unnamed")
    server_type = entry.get("type", "sse")

    if server_type == "sse":
        url = entry["url"]
        logger.info(f"MCP '{name}': SSE -> {url}")
        return MCPServerSSE(url=url)

    if server_type == "stdio":
        command = entry["command"]
        args = entry.get("args", [])
        env = entry.get("env")
        logger.info(f"MCP '{name}': stdio -> {command} {' '.join(args)}")
        return MCPServerStdio(command=command, args=args, env=env)

    logger.warning(f"Unknown MCP server type '{server_type}' for '{name}', skipping")
    return None
