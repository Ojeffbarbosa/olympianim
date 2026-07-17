"""Persistent, non-sensitive defaults for new generation projects."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from olympianim.database.repository import ProjectRepository

_SETTING_KEY = "generation_preferences"


@dataclass(frozen=True)
class GenerationPreferences:
    """Teacher choices reused when starting a future project.

    Credentials are deliberately absent. API keys remain only in Streamlit's
    session state and are never written through this service.
    """

    llm_provider: str = ""
    llm_model: str = ""
    voiceover_enabled: bool = False
    voice_provider: str = ""
    voice_model: str = ""
    voice: str = ""
    voice_language: str = ""
    voice_speed: float = 1.0
    reuse_llm_api_key: bool = False
    color_palette_id: str = ""


class GenerationPreferencesService:
    """Store and retrieve reusable generation choices from ``app_settings``."""

    def __init__(self, repository: ProjectRepository | None = None) -> None:
        self.repository = repository or ProjectRepository()

    def load(self) -> GenerationPreferences | None:
        """Return saved preferences, ignoring malformed persisted values safely."""
        raw_value = self.repository.get_setting(_SETTING_KEY)
        if not raw_value:
            return None
        try:
            payload = json.loads(raw_value)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None

        try:
            return GenerationPreferences(
                llm_provider=self._string(payload, "llm_provider"),
                llm_model=self._string(payload, "llm_model"),
                voiceover_enabled=self._boolean(payload, "voiceover_enabled"),
                voice_provider=self._string(payload, "voice_provider"),
                voice_model=self._string(payload, "voice_model"),
                voice=self._string(payload, "voice"),
                voice_language=self._string(payload, "voice_language"),
                voice_speed=self._speed(payload),
                reuse_llm_api_key=self._boolean(payload, "reuse_llm_api_key"),
                color_palette_id=self._string(payload, "color_palette_id"),
            )
        except (TypeError, ValueError):
            return None

    def save(self, preferences: GenerationPreferences) -> None:
        """Persist only the explicitly supported non-sensitive preference fields."""
        self.repository.set_setting(
            _SETTING_KEY,
            json.dumps(asdict(preferences), ensure_ascii=True, sort_keys=True),
        )

    @staticmethod
    def _string(payload: dict[str, object], key: str) -> str:
        value = payload.get(key, "")
        return value if isinstance(value, str) else ""

    @staticmethod
    def _boolean(payload: dict[str, object], key: str) -> bool:
        value = payload.get(key, False)
        return value if isinstance(value, bool) else False

    @staticmethod
    def _speed(payload: dict[str, object]) -> float:
        value = payload.get("voice_speed", 1.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return 1.0
        speed = float(value)
        if not 0.5 <= speed <= 2.0:
            return 1.0
        return speed
