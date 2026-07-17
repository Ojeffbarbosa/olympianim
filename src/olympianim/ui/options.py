"""Static option lists shown in the Streamlit interface.

The UI keeps only stable voice and language lists here. Provider models
come from the configurable local catalog.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

# Voices available per provider.
VOICE_OPTIONS: Final[Mapping[str, tuple[str, ...]]] = {
    "OpenAI": ("alloy", "echo", "fable", "onyx", "nova", "shimmer", "coral", "sage", "ash"),
    "Google": (
        "Kore",
        "Zephyr",
        "Puck",
        "Charon",
        "Fenrir",
        "Leda",
        "Orus",
        "Aoede",
        "Callirrhoe",
        "Autonoe",
        "Enceladus",
        "Iapetus",
        "Umbriel",
        "Algieba",
        "Despina",
        "Erinome",
        "Algenib",
        "Rasalgethi",
        "Laomedeia",
        "Achernar",
        "Alnilam",
        "Schedar",
        "Gacrux",
        "Pulcherrima",
        "Achird",
        "Zubenelgenubi",
        "Vindemiatrix",
        "Sadachbia",
        "Sadaltager",
        "Sulafat",
    ),
}

# Languages offered for narration. Portuguese is the project default.
VOICE_LANGUAGES: Final[tuple[str, ...]] = (
    "Português (Brasil)",
    "Português (Portugal)",
    "Inglês (EUA)",
    "Espanhol (Espanha)",
)

# ---------------------------------------------------------------------------
# Mathematical area selector
# ---------------------------------------------------------------------------
# The first option lets the IA detect the area automatically; the
# remaining options let the teacher override that detection.
MATH_AREAS: Final[tuple[str, ...]] = (
    "Automática",
    "Geometria",
    "Teoria dos Números",
    "Combinatória",
    "Álgebra",
    "Análise",
    "Probabilidade",
)

DEFAULT_MATH_AREA: Final[str] = "Automática"


def models_for(provider: str) -> tuple[str, ...]:
    """Return the model list associated with ``provider``.

    Falls back to an empty tuple when the provider is unknown so the
    selector degrades gracefully instead of raising.
    """
    from olympianim.services.model_catalog import ModelCatalogService

    return ModelCatalogService().model_ids(provider, "text")


def voice_models_for(provider: str) -> tuple[str, ...]:
    """Return TTS models offered by one voice provider."""
    from olympianim.services.model_catalog import ModelCatalogService

    return ModelCatalogService().model_ids(provider, "speech")


def active_llm_providers() -> tuple[str, ...]:
    from olympianim.services.model_catalog import ModelCatalogService

    return ModelCatalogService().providers("text")


def active_voice_providers() -> tuple[str, ...]:
    from olympianim.services.model_catalog import ModelCatalogService

    return ModelCatalogService().providers("speech")


def model_label(provider: str, model_id: str, *, modality: str = "text") -> str:
    from olympianim.services.model_catalog import ModelCatalogService

    return ModelCatalogService().label(provider, modality, model_id)


def voices_for(provider: str) -> tuple[str, ...]:
    """Return the voice list associated with ``provider``."""
    return VOICE_OPTIONS.get(provider, ())
