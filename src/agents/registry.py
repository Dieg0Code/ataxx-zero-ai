from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agents.heuristic import DEFAULT_HEURISTIC_LEVEL, is_supported_heuristic_level

VariantValidator = Callable[[str], bool]


@dataclass(frozen=True)
class AgentKindRegistration:
    kind: str
    supports_variant: bool = False
    default_variant: str | None = None
    variant_validator: VariantValidator | None = None


@dataclass(frozen=True)
class ParsedAgentSpec:
    kind: str
    variant: str | None = None

    @property
    def canonical(self) -> str:
        if self.variant is None:
            return self.kind
        return f"{self.kind}:{self.variant}"


_AGENT_REGISTRY: dict[str, AgentKindRegistration] = {}


def register_agent_kind(
    *,
    kind: str,
    supports_variant: bool = False,
    default_variant: str | None = None,
    variant_validator: VariantValidator | None = None,
) -> None:
    normalized_kind = kind.strip().lower()
    if normalized_kind == "":
        raise ValueError("Agent kind must not be empty.")
    if not supports_variant and default_variant is not None:
        raise ValueError("Non-variant agent kinds cannot declare a default variant.")
    _AGENT_REGISTRY[normalized_kind] = AgentKindRegistration(
        kind=normalized_kind,
        supports_variant=supports_variant,
        default_variant=default_variant,
        variant_validator=variant_validator,
    )


def registered_agent_kinds() -> tuple[str, ...]:
    return tuple(sorted(_AGENT_REGISTRY.keys()))


def parse_agent_spec(
    agent_spec: str,
    *,
    default_heuristic_level: str = DEFAULT_HEURISTIC_LEVEL,
) -> ParsedAgentSpec:
    normalized_spec = agent_spec.strip().lower()
    if normalized_spec == "":
        raise ValueError("Agent spec must not be empty.")

    kind, has_variant, raw_variant = normalized_spec.partition(":")
    registration = _AGENT_REGISTRY.get(kind)
    if registration is None:
        supported = ", ".join(registered_agent_kinds())
        raise ValueError(f"Unsupported agent kind '{kind}'. Supported kinds: {supported}")

    if not registration.supports_variant:
        if has_variant != "":
            raise ValueError(f"Agent kind '{kind}' does not accept a variant.")
        return ParsedAgentSpec(kind=kind, variant=None)

    resolved_variant = raw_variant.strip()
    if resolved_variant == "":
        if kind == "heuristic":
            resolved_variant = default_heuristic_level.strip().lower()
        elif registration.default_variant is not None:
            resolved_variant = registration.default_variant.strip().lower()
    if resolved_variant == "":
        raise ValueError(f"Agent kind '{kind}' requires a variant.")

    if registration.variant_validator is not None and not registration.variant_validator(resolved_variant):
        raise ValueError(f"Unsupported variant '{resolved_variant}' for agent kind '{kind}'.")

    return ParsedAgentSpec(kind=kind, variant=resolved_variant)


def canonicalize_agent_spec(
    agent_spec: str,
    *,
    default_heuristic_level: str = DEFAULT_HEURISTIC_LEVEL,
) -> str:
    return parse_agent_spec(
        agent_spec,
        default_heuristic_level=default_heuristic_level,
    ).canonical


def agent_kind(
    agent_spec: str,
    *,
    default_heuristic_level: str = DEFAULT_HEURISTIC_LEVEL,
) -> str:
    return parse_agent_spec(
        agent_spec,
        default_heuristic_level=default_heuristic_level,
    ).kind


def heuristic_level_for_agent(
    agent_spec: str,
    *,
    default_heuristic_level: str = DEFAULT_HEURISTIC_LEVEL,
) -> str | None:
    parsed = parse_agent_spec(
        agent_spec,
        default_heuristic_level=default_heuristic_level,
    )
    if parsed.kind != "heuristic":
        return None
    return parsed.variant


register_agent_kind(kind="human")
register_agent_kind(kind="random")
register_agent_kind(kind="model")
register_agent_kind(
    kind="heuristic",
    supports_variant=True,
    default_variant=DEFAULT_HEURISTIC_LEVEL,
    variant_validator=is_supported_heuristic_level,
)


__all__ = [
    "ParsedAgentSpec",
    "agent_kind",
    "canonicalize_agent_spec",
    "heuristic_level_for_agent",
    "parse_agent_spec",
    "register_agent_kind",
    "registered_agent_kinds",
]
