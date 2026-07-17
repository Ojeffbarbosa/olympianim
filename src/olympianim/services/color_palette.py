"""Configurable semantic color palettes for generated Manim scenes."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, dataclass, replace

from olympianim.database.models import ColorPaletteRecord
from olympianim.database.repository import ProjectRepository, new_id, utc_now

_DEFAULTS_VERSION = "1"
_HEX_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")
_COLOR_FIELDS = (
    "background",
    "primary_text",
    "secondary_text",
    "surface",
    "primary",
    "secondary",
    "highlight",
    "stroke",
)


@dataclass(frozen=True)
class ColorPaletteInput:
    """Editable semantic colors accepted by the palette catalog."""

    name: str
    description: str = ""
    background: str = "#000000"
    primary_text: str = "#FFFFFF"
    secondary_text: str = "#BBBBBB"
    surface: str = "#222222"
    primary: str = "#58C4DD"
    secondary: str = "#83C167"
    highlight: str = "#FFFF00"
    stroke: str = "#FFFFFF"
    enabled: bool = True
    sort_order: int = 0


@dataclass(frozen=True)
class _BuiltinPalette(ColorPaletteInput):
    stable_id: str = ""


_BUILTINS = (
    _BuiltinPalette(
        stable_id="builtin:manim-dark",
        name="Manim escura",
        description="Base escura com as cores clássicas do Manim.",
        background="#000000",
        primary_text="#FFFFFF",
        secondary_text="#BBBBBB",
        surface="#222222",
        primary="#58C4DD",
        secondary="#83C167",
        highlight="#FFFF00",
        stroke="#FFFFFF",
        sort_order=0,
    ),
    _BuiltinPalette(
        stable_id="builtin:manim-light",
        name="Manim clara",
        description="Base clara e contrastante com cores do Manim.",
        background="#FFFFFF",
        primary_text="#000000",
        secondary_text="#595959",
        surface="#EEEEEE",
        primary="#236B8E",
        secondary="#699C52",
        highlight="#D55E00",
        stroke="#000000",
        sort_order=1,
    ),
    _BuiltinPalette(
        stable_id="builtin:okabe-ito-dark",
        name="Okabe-Ito escura",
        description="Base escura com acentos distinguíveis por daltonismo.",
        background="#000000",
        primary_text="#FFFFFF",
        secondary_text="#BBBBBB",
        surface="#222222",
        primary="#56B4E9",
        secondary="#009E73",
        highlight="#E69F00",
        stroke="#FFFFFF",
        sort_order=2,
    ),
    _BuiltinPalette(
        stable_id="builtin:okabe-ito-light",
        name="Okabe-Ito clara",
        description="Base clara com acentos distinguíveis por daltonismo.",
        background="#FFFFFF",
        primary_text="#000000",
        secondary_text="#595959",
        surface="#F2F2F2",
        primary="#0072B2",
        secondary="#009E73",
        highlight="#D55E00",
        stroke="#000000",
        sort_order=3,
    ),
)


class ColorPaletteService:
    """Own palette defaults, validation, snapshots and prompt context."""

    def __init__(self, repository: ProjectRepository | None = None) -> None:
        self.repository = repository or ProjectRepository()

    def ensure_defaults(self) -> None:
        """Seed or refresh protected built-ins without touching custom palettes."""
        version = self.repository.get_setting("color_palette_defaults_version")
        existing = {item.id: item for item in self.repository.list_color_palettes()}
        if version == _DEFAULTS_VERSION and all(item.stable_id in existing for item in _BUILTINS):
            return
        now = utc_now()
        for item in _BUILTINS:
            current = existing.get(item.stable_id)
            self.repository.save_color_palette(
                ColorPaletteRecord(
                    id=item.stable_id,
                    name=item.name,
                    description=item.description,
                    background=item.background,
                    primary_text=item.primary_text,
                    secondary_text=item.secondary_text,
                    surface=item.surface,
                    primary=item.primary,
                    secondary=item.secondary,
                    highlight=item.highlight,
                    stroke=item.stroke,
                    enabled=True,
                    is_builtin=True,
                    revision=(current.revision + 1) if current else 1,
                    sort_order=item.sort_order,
                    created_at=current.created_at if current else now,
                    updated_at=now,
                )
            )
        self.repository.set_setting("color_palette_defaults_version", _DEFAULTS_VERSION)

    def list_palettes(self, *, enabled_only: bool = False) -> list[ColorPaletteRecord]:
        self.ensure_defaults()
        return self.repository.list_color_palettes(enabled_only=enabled_only)

    def get(self, palette_id: str) -> ColorPaletteRecord | None:
        if not palette_id:
            return None
        self.ensure_defaults()
        return self.repository.get_color_palette(palette_id)

    def save(
        self,
        data: ColorPaletteInput,
        *,
        record_id: str | None = None,
    ) -> ColorPaletteRecord:
        """Create or update a custom palette after validating contrast."""
        self._validate(data)
        current = self.repository.get_color_palette(record_id) if record_id else None
        if current and current.is_builtin:
            raise ValueError("Duplique a paleta fornecida pelo app antes de editá-la.")
        now = utc_now()
        record = ColorPaletteRecord(
            id=current.id if current else new_id(),
            **asdict(data),
            is_builtin=False,
            revision=current.revision + 1 if current else 1,
            created_at=current.created_at if current else now,
            updated_at=now,
        )
        try:
            return self.repository.save_color_palette(record)
        except sqlite3.IntegrityError as exc:
            raise ValueError("Já existe uma paleta com esse nome.") from exc

    def duplicate(self, palette_id: str) -> ColorPaletteRecord:
        """Create an editable copy of any catalog palette."""
        source = self._required(palette_id)
        names = {item.name for item in self.list_palettes()}
        base_name = f"{source.name} - cópia"
        name = base_name
        suffix = 2
        while name in names:
            name = f"{base_name} {suffix}"
            suffix += 1
        return self.save(
            ColorPaletteInput(
                name=name,
                description=source.description,
                **{field: getattr(source, field) for field in _COLOR_FIELDS},
                sort_order=source.sort_order + 1,
            )
        )

    def deactivate(self, palette_id: str) -> ColorPaletteRecord:
        """Hide a custom palette from new-project selectors."""
        current = self._required(palette_id)
        if current.is_builtin:
            raise ValueError("Paletas fornecidas pelo app não podem ser desativadas.")
        return self.repository.save_color_palette(
            replace(current, enabled=False, updated_at=utc_now())
        )

    def snapshot(self, palette_id: str) -> str:
        """Serialize a palette so a project remains reproducible."""
        palette = self.get(palette_id)
        if palette is None or not palette.enabled:
            return ""
        payload = {"id": palette.id, "name": palette.name}
        payload.update({field: getattr(palette, field) for field in _COLOR_FIELDS})
        return json.dumps(payload, ensure_ascii=True, sort_keys=True)

    @staticmethod
    def prompt_context(snapshot: str) -> str:
        """Build compact builder guidance; return nothing for automatic mode."""
        if not snapshot:
            return ""
        try:
            data = json.loads(snapshot)
        except (json.JSONDecodeError, TypeError):
            return ""
        if not isinstance(data, dict) or not all(field in data for field in _COLOR_FIELDS):
            return ""
        labels = {
            "background": "fundo",
            "primary_text": "texto principal",
            "secondary_text": "texto secundário",
            "surface": "superfície",
            "primary": "primária",
            "secondary": "secundária",
            "highlight": "destaque",
            "stroke": "contorno",
        }
        colors = ", ".join(f"{labels[field]} {data[field]}" for field in _COLOR_FIELDS)
        return (
            f"Paleta visual selecionada: {colors}. "
            "Use cada cor conforme sua função, preserve contraste local e não dependa "
            "somente da cor para comunicar significado."
        )

    @classmethod
    def _validate(cls, data: ColorPaletteInput) -> None:
        if not data.name.strip():
            raise ValueError("Informe o nome da paleta.")
        for field in _COLOR_FIELDS:
            value = getattr(data, field)
            if not _HEX_PATTERN.fullmatch(value):
                raise ValueError(f"A cor {field} deve usar o formato #RRGGBB.")
        for text_field in ("primary_text", "secondary_text"):
            for base_field in ("background", "surface"):
                ratio = cls._contrast(getattr(data, text_field), getattr(data, base_field))
                if ratio < 4.5:
                    raise ValueError(
                        "Textos principal e secundário precisam de contraste mínimo "
                        "de 4.5:1 sobre fundo e superfície."
                    )
        if cls._contrast(data.stroke, data.background) < 3.0:
            raise ValueError("O contorno precisa de contraste mínimo de 3:1 com o fundo.")

    @staticmethod
    def _contrast(first: str, second: str) -> float:
        def luminance(color: str) -> float:
            channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
            linear = [
                channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
                for channel in channels
            ]
            return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

        lighter, darker = sorted((luminance(first), luminance(second)), reverse=True)
        return (lighter + 0.05) / (darker + 0.05)

    def _required(self, palette_id: str) -> ColorPaletteRecord:
        palette = self.get(palette_id)
        if palette is None:
            raise ValueError("Paleta não encontrada.")
        return palette
