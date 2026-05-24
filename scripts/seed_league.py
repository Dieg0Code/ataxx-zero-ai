"""Materializa una semilla de league en checkpoints/league_ratings.json.

Uso típico (Kaggle setup, antes de train.py):
    python scripts/seed_league.py --seed seeds/league_v13_seed.json

Idempotente: si el destino ya existe, no sobreescribe (a menos que
se pase --force). Verifica que cada codename tenga su .pt presente
en checkpoints/registered/<codename>.pt; entradas faltantes se
omiten con warning (la league quedará más chica pero funcional).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, help="Path al JSON de semilla.")
    parser.add_argument(
        "--out",
        default="checkpoints/league_ratings.json",
        help="Destino del league state materializado.",
    )
    parser.add_argument(
        "--registered-dir",
        default="checkpoints/registered",
        help="Directorio donde buscar <codename>.pt.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescribe el destino aunque ya exista.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seed_path = Path(args.seed)
    out_path = Path(args.out)
    registered_dir = Path(args.registered_dir)

    if out_path.exists() and not args.force:
        print(f"[seed_league] {out_path} ya existe; usar --force para sobrescribir.")
        return

    seed_payload = json.loads(seed_path.read_text(encoding="utf-8"))
    raw_entries = seed_payload.get("entries", [])
    if not isinstance(raw_entries, list):
        raise ValueError("Seed JSON: 'entries' debe ser lista.")

    entries: dict[str, dict[str, str | float | int]] = {}
    skipped: list[str] = []
    for raw in raw_entries:
        if not isinstance(raw, dict):
            continue
        codename = str(raw["codename"])
        artifact_path = registered_dir / f"{codename}.pt"
        if not artifact_path.exists():
            skipped.append(codename)
            continue
        participant_id = f"ckpt:{codename}"
        entries[participant_id] = {
            "participant_id": participant_id,
            "display_name": codename,
            "kind": "checkpoint",
            "rating": float(raw.get("rating", 1200.0)),
            "games_played": int(raw.get("games_played", 4)),
            "artifact_path": str(artifact_path).replace("\\", "/"),
        }

    champion_id = seed_payload.get("champion_id")
    if not isinstance(champion_id, str) or champion_id not in entries:
        # Fallback: el de mayor rating.
        champion_id = max(entries.values(), key=lambda e: float(e["rating"]))["participant_id"] if entries else None

    league_state = {
        "version": int(seed_payload.get("version", 1)),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "champion_id": champion_id,
        "entries": entries,
        "recent_series": [],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(league_state, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[seed_league] escribió {out_path} con {len(entries)} entradas; champion={champion_id}")
    if skipped:
        print(f"[seed_league] omitidos por falta de .pt en {registered_dir}: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
