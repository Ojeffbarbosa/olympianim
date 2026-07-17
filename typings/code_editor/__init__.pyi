from typing import Any

def code_editor(
    code: str,
    *,
    lang: str,
    height: int | str | tuple[int, int] | list[int],
    key: str,
    response_mode: str | tuple[str, ...] = ...,
    buttons: list[dict[str, Any]] | None = ...,
    menu: dict[str, Any] | None = ...,
    props: dict[str, Any] | None = ...,
    options: dict[str, Any] | None = ...,
    component_props: dict[str, Any] | None = ...,
) -> dict[str, Any]: ...
