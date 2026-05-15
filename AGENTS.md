# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.10+ AlphaZero-style Ataxx engine. Core code lives in `src/`:

- `src/game/`: board representation, rules, actions, serialization, constants.
- `src/engine/`: MCTS search.
- `src/model/`: neural network, checkpoint compatibility, model registry.
- `src/training/`: self-play, curriculum, evaluation, checkpointing, training loop helpers.
- `src/inference/`: model loading and runtime services.
- `src/agents/`: random, heuristic, model, and selector agents.
- `src/data/`: replay buffer and datasets.
- `src/ui/arena/`: Pygame arena, HUD, replay recording, assets.

Tests are in `tests/`. Operational scripts live in `scripts/`; entry points include `train.py`, `train_improved.py`, and `main.py`. Treat `checkpoints/`, `runs_history/`, `arena_screenshots/`, and `tournament_replays/` as generated artifacts.

## Build, Test, and Development Commands

- `uv sync --all-groups`: install runtime, dev, UI, training, export, and inference dependencies.
- `uv run pytest -q`: run the full test suite.
- `uv run ruff check train.py src/engine src/model src/game src/data tests scripts`: run the same lint scope used by CI.
- `uv run pyrefly check train.py src tests`: run static type checks.
- `uv run python scripts/check_python_max_lines.py --max-lines 500 --path src --path tests --path scripts --path train.py`: enforce file-length policy.
- `uv run python train.py --iterations 2 --episodes 8 --epochs 1 --sims 80 --batch-size 64 --save-every 1 --verbose`: quick training smoke test.
- `uv run python scripts/play_pygame.py --mode play --opponent heuristic --level hard`: launch the local arena.

## Coding Style & Naming Conventions

Use 4-space indentation, double quotes, and Ruff formatting conventions. Ruff targets Python 3.10 with an 88-character line length; `E501` is ignored, but keep lines readable. Prefer typed functions in library code. Keep Python files under 500 lines.

Name modules and functions in `snake_case`, classes in `PascalCase`, constants in `UPPER_SNAKE_CASE`, and tests as `test_<behavior>.py`.

Reuse existing systems before adding parallel implementations. In particular, UI work in `src/ui/arena/` should reuse the arena renderer, HUD, layout, and theme when it is showing the same game board or pieces; only add specialized overlays or state where the behavior genuinely differs.

Do not assume local developer artifacts exist in Kaggle, CI, or other clean runtimes. Any code that resolves checkpoints, model aliases, replay data, or registries must use the repository's existing resolver/checkpointer paths, and tests must cover the clean-runtime case where local `checkpoints/registry.json` is absent. Avoid one-off path lookups when an existing registry, HuggingFace, or runtime helper already owns that concern.

## Testing Guidelines

Use pytest/unittest-compatible tests under `tests/`. Add focused tests near the affected subsystem: rules in `test_board_rules.py`, MCTS in `test_mcts_numerics.py`, training behavior in `test_training_*.py`, inference in `test_inference_*.py`, and arena runtime in `test_ui_arena_model_runtime.py`. For numerical or training changes, use deterministic seeds or small fixtures.

## Commit & Pull Request Guidelines

Recent commits use short, imperative summaries such as `implement tournament`, `fix CI: ...`, and `arena: ...`. Keep the subject specific and mention the subsystem when helpful.

PRs should describe the behavior change, list validation commands run, link related issues or runs, and include screenshots only for arena/UI changes. Avoid bundling generated checkpoints, replay dumps, screenshots, or temporary Kaggle/HuggingFace output.
