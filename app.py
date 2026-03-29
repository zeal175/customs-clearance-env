"""
Hugging Face Docker Spaces expect an application module (often `app.py` / `app:app`).
All routes and logic live in `main.py`; this file only re-exports the ASGI app.
"""

from main import app

__all__ = ["app"]
