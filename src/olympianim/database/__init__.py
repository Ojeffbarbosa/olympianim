"""Database layer: SQLite models, repository access and connection helpers.

This module must never store API keys.
"""

from olympianim.database.models import (
    GeneratedFileRecord,
    GenerationLogRecord,
    ProjectCreate,
    ProjectPromptRecord,
    ProjectRecord,
    PromptRecord,
    PromptVersionRecord,
)
from olympianim.database.repository import ProjectRepository
from olympianim.database.sqlite import connect, initialize_database

__all__ = [
    "GeneratedFileRecord",
    "GenerationLogRecord",
    "ProjectCreate",
    "ProjectPromptRecord",
    "ProjectRecord",
    "ProjectRepository",
    "PromptRecord",
    "PromptVersionRecord",
    "connect",
    "initialize_database",
]
