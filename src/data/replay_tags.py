from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReplayTag:
    value: str
    label: str
    training_role: str


REPLAY_TAGS = (
    ReplayTag("", "sin tag", "neutral"),
    ReplayTag("human_instructive", "humana util", "prioritize"),
    ReplayTag("brilliant", "brillante", "prioritize"),
    ReplayTag("late_game", "late game", "prioritize"),
    ReplayTag("ai_mistake", "error IA", "neutral"),
    ReplayTag("demo", "demo", "prioritize"),
    ReplayTag("review", "revisar", "neutral"),
    ReplayTag("discard", "descartar", "exclude"),
)

LEGACY_TAG_ALIASES = {
    "good": "human_instructive",
    "bad": "discard",
    "bug": "discard",
}

TAG_VALUES = tuple(tag.value for tag in REPLAY_TAGS)
TAG_LABELS = {tag.value: tag.label for tag in REPLAY_TAGS}
TAG_ROLES = {tag.value: tag.training_role for tag in REPLAY_TAGS}


def normalize_replay_tag(value: object) -> str:
    if not isinstance(value, str):
        return ""
    clean = value.strip().lower()
    return LEGACY_TAG_ALIASES.get(clean, clean if clean in TAG_VALUES else "")


def replay_tag_label(value: object) -> str:
    return TAG_LABELS.get(normalize_replay_tag(value), "sin tag")


__all__ = [
    "LEGACY_TAG_ALIASES",
    "REPLAY_TAGS",
    "TAG_LABELS",
    "TAG_ROLES",
    "TAG_VALUES",
    "ReplayTag",
    "normalize_replay_tag",
    "replay_tag_label",
]
