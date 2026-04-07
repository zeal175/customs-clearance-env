"""
FastAPI server for customs-clearance-env (CHA OpenEnv).
Adapts the environment to the OpenEnv SDK for both REST and WebSocket support.
"""
from __future__ import annotations

from fastapi import FastAPI
from openenv.core.env_server.http_server import HTTPEnvServer
from environment_openenv import ChaOpenEnvEnvironment

app = FastAPI(
    title="customs-clearance-env",
    description="CHA (Custom House Agent) sea-freight document processing — OpenEnv-style API",
    version="1.0.0",
)

# Instantiate the environment
env = ChaOpenEnvEnvironment()

# Register OpenEnv routes (REST + WebSocket)
# This handles /reset, /step, /metadata, /health, /schema, etc.
server = HTTPEnvServer(app, env)

# Export app for app.py
__all__ = ["app"]
