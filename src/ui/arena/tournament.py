"""Tournament state para el evento Humanos vs IA de la clase 02 semana 09.

Pure logic, no Pygame dependency. Persistible a JSON para que un crash o un
relaunch puedan retomar el torneo donde quedo.

Reglamento (flyer oficial comunicado a los alumnos):
- N jugadores juegan 2 rondas, cada ronda es Bo3 (3 partidas).
- Misma config para todos: mismo modelo, mismos sims.
- Sorteo de inicio por Bo3: partida 0 sorteo, partida 1 invierte, partida 2 sorteo.
- Puntaje por partida: V=1.0 / E=0.5 / D=0.
- Ranking: puntaje total -> victorias -> diferencia de piezas -> partida relampago.
- Abandonar el torneo: las partidas restantes cuentan como derrota.
- Bonus a la evaluacion final del 19/05: 1+1.0, 2+0.5, 3+0.3.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Bonus para los tres primeros lugares del torneo (sobre la evaluacion final).
RANK_BONUSES: tuple[float, ...] = (1.0, 0.5, 0.3)

DEFAULT_SAVE_PATH = Path.home() / ".ataxx_tournament.json"

POINTS_PER_RESULT: dict[str, float] = {"W": 1.0, "D": 0.5, "L": 0.0}

DEFAULT_ROUNDS = 2
DEFAULT_BO_SIZE = 3
MAX_TIEBREAK_ROUNDS = 3


@dataclass
class MatchRecord:
    """Una partida del torneo: posicion fija dentro del Bo3 + sorteo de inicio."""

    player: str
    round_idx: int                # 0..rounds-1 (regular) ; -1 para partida de desempate
    match_idx: int                # 0..bo_size-1 (regular) ; 0..N para tiebreak iter
    starter: str                  # "human" | "ai" (sorteado al crear el state)
    status: str = "pending"       # pending | completed | forfeit
    result: str = ""              # "W" | "L" | "D" desde la perspectiva humana
    human_pieces: int = 0
    ai_pieces: int = 0
    piece_diff: int = 0           # human_pieces - ai_pieces
    halfmoves: int = 0
    played_at: float = 0.0
    is_tiebreak: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MatchRecord:
        return cls(
            player=str(payload["player"]),
            round_idx=int(payload.get("round_idx", 0)),
            match_idx=int(payload.get("match_idx", 0)),
            starter=str(payload.get("starter", "human")),
            status=str(payload.get("status", "pending")),
            result=str(payload.get("result", "")),
            human_pieces=int(payload.get("human_pieces", 0)),
            ai_pieces=int(payload.get("ai_pieces", 0)),
            piece_diff=int(payload.get("piece_diff", 0)),
            halfmoves=int(payload.get("halfmoves", 0)),
            played_at=float(payload.get("played_at", 0.0)),
            is_tiebreak=bool(payload.get("is_tiebreak", False)),
        )


@dataclass(frozen=True)
class PlayerStanding:
    """Fila del ranking calculada en runtime a partir de los matches."""

    player: str
    points: float
    wins: int
    draws: int
    losses: int
    piece_diff: int
    played: int     # matches con status completed o forfeit


@dataclass
class TournamentConfig:
    """Configuracion del torneo. Inmutable una vez creado el state."""

    players: tuple[str, ...]
    model_label: str = "liga"
    mcts_sims: int = 400
    rounds: int = DEFAULT_ROUNDS
    bo_size: int = DEFAULT_BO_SIZE
    save_path: Path = DEFAULT_SAVE_PATH

    def to_dict(self) -> dict[str, Any]:
        return {
            "players": list(self.players),
            "model_label": self.model_label,
            "mcts_sims": int(self.mcts_sims),
            "rounds": int(self.rounds),
            "bo_size": int(self.bo_size),
            "save_path": str(self.save_path),
        }


def _draw_starters(rng: random.Random, bo_size: int) -> list[str]:
    """Aplica la regla del flyer: partida 0 sorteo, 1 invierte, 2 sorteo, etc."""
    starters: list[str] = []
    for idx in range(bo_size):
        if idx % 2 == 0:
            starters.append("human" if rng.random() < 0.5 else "ai")
        else:
            prev = starters[-1]
            starters.append("ai" if prev == "human" else "human")
    return starters


@dataclass
class TournamentState:
    """Maquina de estado del torneo: roster, orden, matches y persistencia."""

    config: TournamentConfig
    tournament_id: str
    present: list[str] = field(default_factory=list)
    order: list[str] = field(default_factory=list)
    matches: list[MatchRecord] = field(default_factory=list)
    finished: bool = False

    # ----- Construccion -----

    @classmethod
    def new(
        cls,
        config: TournamentConfig,
        *,
        today: str | None = None,
        seed: int | None = None,
    ) -> TournamentState:
        """Crea un torneo nuevo con `rounds * bo_size` partidas por jugador.

        El sorteo de starter es reproducible: usa una semilla derivada del
        tournament_id (la fecha) o de `seed` si se pasa explicitamente.
        """
        stamp = today or time.strftime("%Y-%m-%d")
        rng_seed: int = seed if seed is not None else _stable_seed(stamp)
        # Sorteo del torneo: reproducible, no es uso criptografico.
        rng = random.Random(rng_seed)  # noqa: S311
        matches: list[MatchRecord] = []
        for player in config.players:
            for round_idx in range(config.rounds):
                starters = _draw_starters(rng, config.bo_size)
                for match_idx, starter in enumerate(starters):
                    matches.append(
                        MatchRecord(
                            player=player,
                            round_idx=round_idx,
                            match_idx=match_idx,
                            starter=starter,
                        ),
                    )
        return cls(
            config=config,
            tournament_id=stamp,
            present=list(config.players),
            order=[],
            matches=matches,
        )

    # ----- Roster -----

    def set_present(self, present: list[str]) -> None:
        """Marca quienes estan presentes; los ausentes pasan a forfeit (regla 9)."""
        present_set = {p for p in present if p in self.config.players}
        self.present = [p for p in self.config.players if p in present_set]
        for match in self.matches:
            if match.status == "completed":
                continue
            if match.is_tiebreak:
                continue
            if match.player not in present_set:
                self._apply_forfeit(match)
            else:
                match.status = "pending"

    # ----- Sorteo del orden de jugadores -----

    def draw_order(self, rng: random.Random | None = None) -> list[str]:
        rng = rng or random.Random()  # noqa: S311
        order = list(self.present)
        rng.shuffle(order)
        self.order = order
        return list(order)

    # ----- Flujo de matches -----

    def next_pending_match(self) -> MatchRecord | None:
        """Devuelve el proximo match pendiente respetando ronda -> jugador -> partida.

        Orden: ronda 0 completa (jugador A Bo3 -> jugador B Bo3 -> ...) y recien
        despues ronda 1. Asi el torneo se siente "se completa una ronda y empieza
        la siguiente" en vez de "cada jugador agota sus 6 partidas seguidas".
        """
        if not self.order:
            return None
        for round_idx in range(self.config.rounds):
            for name in self.order:
                for match in self._matches_for(name):
                    if match.round_idx != round_idx:
                        continue
                    if match.status == "pending":
                        return match
        return None

    def record_result(
        self,
        match: MatchRecord,
        *,
        human_pieces: int,
        ai_pieces: int,
        halfmoves: int,
    ) -> MatchRecord:
        match.human_pieces = int(human_pieces)
        match.ai_pieces = int(ai_pieces)
        match.piece_diff = int(human_pieces) - int(ai_pieces)
        match.halfmoves = int(halfmoves)
        if human_pieces > ai_pieces:
            match.result = "W"
        elif human_pieces < ai_pieces:
            match.result = "L"
        else:
            match.result = "D"
        match.status = "completed"
        match.played_at = time.time()
        return match

    def forfeit_remaining_for(self, player: str) -> int:
        """Marca todas las partidas pendientes del jugador como derrota (regla 9).

        Devuelve cuantas partidas se aplicaron.
        """
        count = 0
        for match in self._matches_for(player):
            if match.status == "pending":
                self._apply_forfeit(match)
                count += 1
        return count

    def finish(self) -> None:
        """Cierra el torneo: las pendientes restantes pasan a forfeit (regla 9)."""
        for match in self.matches:
            if match.status == "pending":
                self._apply_forfeit(match)
        self.finished = True

    # ----- Scoring agregado -----

    def standing_for(self, player: str) -> PlayerStanding:
        wins = draws = losses = played = 0
        diff = 0
        points = 0.0
        for match in self._matches_for(player):
            if match.status not in {"completed", "forfeit"}:
                continue
            played += 1
            points += POINTS_PER_RESULT.get(match.result, 0.0)
            diff += match.piece_diff
            if match.result == "W":
                wins += 1
            elif match.result == "D":
                draws += 1
            elif match.result == "L":
                losses += 1
        return PlayerStanding(
            player=player,
            points=points,
            wins=wins,
            draws=draws,
            losses=losses,
            piece_diff=diff,
            played=played,
        )

    def ranking(self) -> list[PlayerStanding]:
        """Ranking respetando los 3 criterios del flyer.

        Solo incluye jugadores con al menos una partida registrada (completed o
        forfeit). El desempate de relampago, si hay, se aplica sobre el orden
        derivado de estos 3 criterios. Empates persistentes quedan en el orden
        de aparicion en `self.order`.
        """
        standings = [self.standing_for(p) for p in self.config.players]
        # Solo cuentan jugadores que registraron al menos un resultado.
        standings = [s for s in standings if s.played > 0]
        standings.sort(
            key=lambda s: (s.points, s.wins, s.piece_diff),
            reverse=True,
        )
        # Aplica el desempate de relampago, si hubo, para reordenar pares empatados.
        return self._apply_tiebreaks(standings)

    def bonus_for_rank(self, rank_idx: int) -> float:
        """Bonus posicional (0-based) que aplica sobre la evaluacion final."""
        if 0 <= rank_idx < len(RANK_BONUSES):
            return RANK_BONUSES[rank_idx]
        return 0.0

    # ----- Tiebreak (partida relampago) -----

    def needs_tiebreak(self) -> tuple[str, str] | None:
        """Detecta empate persistente en los 3 criterios entre dos posiciones del top 3.

        Solo manejamos empates de a 2 (caso esperado). Empates de 3+ se dejan
        en el podio con nota; no devolvemos nada porque no podemos resolverlos
        con una sola partida relampago entre dos jugadores.
        """
        # Standings sin aplicar tiebreaks previos: medimos sobre los criterios base.
        base = [self.standing_for(p) for p in self.config.players]
        base = [s for s in base if s.played > 0]
        base.sort(key=lambda s: (s.points, s.wins, s.piece_diff), reverse=True)
        if len(base) < 2:
            return None

        def tied(a: PlayerStanding, b: PlayerStanding) -> bool:
            return (a.points, a.wins, a.piece_diff) == (b.points, b.wins, b.piece_diff)

        for idx in range(min(3, len(base)) - 1):
            a = base[idx]
            b = base[idx + 1]
            if not tied(a, b):
                continue
            # Si ya hay un tiebreak resuelto entre ese par, no volver a pedirlo.
            if self._tiebreak_resolved_between(a.player, b.player):
                continue
            # Si hay un tiebreak en curso (a la espera de jugar), tampoco.
            if self._tiebreak_pending_between(a.player, b.player):
                continue
            # Cap de iteraciones para no loopear si ambos siguen empatados.
            if self._tiebreak_round_count(a.player, b.player) >= MAX_TIEBREAK_ROUNDS:
                continue
            return (a.player, b.player)
        return None

    def add_tiebreak_pair(
        self,
        player_a: str,
        player_b: str,
        *,
        rng: random.Random | None = None,
    ) -> tuple[MatchRecord, MatchRecord]:
        """Crea un par de matches de desempate (uno por jugador), starter sorteado."""
        rng = rng or random.Random()  # noqa: S311
        iteration = self._tiebreak_round_count(player_a, player_b)
        records: list[MatchRecord] = []
        for player in (player_a, player_b):
            starter = "human" if rng.random() < 0.5 else "ai"
            rec = MatchRecord(
                player=player,
                round_idx=-1,
                match_idx=iteration,
                starter=starter,
                is_tiebreak=True,
            )
            self.matches.append(rec)
            records.append(rec)
        return records[0], records[1]

    def next_pending_tiebreak(self) -> MatchRecord | None:
        for match in self.matches:
            if match.is_tiebreak and match.status == "pending":
                return match
        return None

    # ----- Helpers internos -----

    def _matches_for(self, player: str) -> list[MatchRecord]:
        return [
            m for m in self.matches
            if m.player == player and not m.is_tiebreak
        ]

    def _tiebreak_matches_between(self, a: str, b: str) -> list[MatchRecord]:
        return [
            m for m in self.matches
            if m.is_tiebreak and m.player in {a, b}
        ]

    def _tiebreak_round_count(self, a: str, b: str) -> int:
        recs = self._tiebreak_matches_between(a, b)
        # Cada iteracion son 2 matches (uno por jugador).
        return len(recs) // 2

    def _tiebreak_pending_between(self, a: str, b: str) -> bool:
        return any(
            m.status == "pending" for m in self._tiebreak_matches_between(a, b)
        )

    def _tiebreak_resolved_between(self, a: str, b: str) -> bool:
        """True si los ultimos tiebreaks entre A y B definieron un ganador."""
        recs = self._tiebreak_matches_between(a, b)
        if not recs:
            return False
        # Ultima iteracion completa (par completo, ambos completed).
        completed = [m for m in recs if m.status == "completed"]
        if len(completed) < 2:
            return False
        # Verifica que la ultima iteracion (max match_idx) tenga ambos jugadores.
        last_iter = max(m.match_idx for m in completed)
        last_recs = [m for m in completed if m.match_idx == last_iter]
        if len(last_recs) < 2:
            return False
        return self._tiebreak_winner(last_recs) is not None

    def _tiebreak_winner(self, pair: list[MatchRecord]) -> str | None:
        """Compara dos MatchRecord de tiebreak y devuelve el ganador, o None si empatan."""
        if len(pair) != 2:
            return None
        ranking = {"W": 2, "D": 1, "L": 0}
        a, b = pair
        sa, sb = ranking.get(a.result, 0), ranking.get(b.result, 0)
        if sa != sb:
            return a.player if sa > sb else b.player
        # Mismo resultado: desempate por piece_diff de la propia relampago.
        if a.piece_diff != b.piece_diff:
            return a.player if a.piece_diff > b.piece_diff else b.player
        return None

    def _apply_tiebreaks(self, standings: list[PlayerStanding]) -> list[PlayerStanding]:
        """Reordena pares contiguos empatados segun el resultado del relampago."""
        if len(standings) < 2:
            return standings
        result = list(standings)
        idx = 0
        while idx < len(result) - 1:
            a, b = result[idx], result[idx + 1]
            if (a.points, a.wins, a.piece_diff) != (b.points, b.wins, b.piece_diff):
                idx += 1
                continue
            recs = self._tiebreak_matches_between(a.player, b.player)
            completed = [m for m in recs if m.status == "completed"]
            if len(completed) < 2:
                idx += 1
                continue
            last_iter = max(m.match_idx for m in completed)
            last_recs = [m for m in completed if m.match_idx == last_iter]
            winner = self._tiebreak_winner(last_recs)
            if winner is None:
                idx += 1
                continue
            if winner == b.player:
                # b debe ir antes que a.
                result[idx], result[idx + 1] = result[idx + 1], result[idx]
            idx += 1
        return result

    def _apply_forfeit(self, match: MatchRecord) -> None:
        match.status = "forfeit"
        match.result = "L"
        match.human_pieces = 0
        match.ai_pieces = 0
        match.piece_diff = 0
        match.halfmoves = 0
        match.played_at = time.time()

    # ----- Acceso publico de lectura -----

    def matches_for(self, player: str) -> list[MatchRecord]:
        return list(self._matches_for(player))

    def regular_matches(self) -> list[MatchRecord]:
        return [m for m in self.matches if not m.is_tiebreak]

    def tiebreak_matches(self) -> list[MatchRecord]:
        return [m for m in self.matches if m.is_tiebreak]

    def total_active(self) -> int:
        """Cantidad de matches regulares activos (no tiebreak)."""
        return len(self.regular_matches())

    def total_completed_regular(self) -> int:
        return sum(
            1 for m in self.regular_matches()
            if m.status in {"completed", "forfeit"}
        )

    # ----- Persistencia -----

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "tournament_id": self.tournament_id,
            "present": list(self.present),
            "order": list(self.order),
            "matches": [m.to_dict() for m in self.matches],
            "finished": self.finished,
        }

    def persist(self, path: Path | None = None) -> None:
        target = Path(path) if path is not None else self.config.save_path
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(self.to_dict(), indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

    @classmethod
    def load(cls, path: Path | None = None) -> TournamentState | None:
        source = Path(path) if path is not None else DEFAULT_SAVE_PATH
        if not source.is_file():
            return None
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        config_payload = payload.get("config", {})
        config = TournamentConfig(
            players=tuple(config_payload.get("players", [])),
            model_label=str(config_payload.get("model_label", "liga")),
            mcts_sims=int(config_payload.get("mcts_sims", 400)),
            rounds=int(config_payload.get("rounds", DEFAULT_ROUNDS)),
            bo_size=int(config_payload.get("bo_size", DEFAULT_BO_SIZE)),
            save_path=Path(config_payload.get("save_path", str(source))),
        )
        return cls(
            config=config,
            tournament_id=str(payload.get("tournament_id", "")),
            present=list(payload.get("present", [])),
            order=list(payload.get("order", [])),
            matches=[MatchRecord.from_dict(m) for m in payload.get("matches", [])],
            finished=bool(payload.get("finished", False)),
        )


def _stable_seed(text: str) -> int:
    """Hash determinista (no usa hash() porque PYTHONHASHSEED lo aleatoriza)."""
    h = 0
    for ch in text:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return int(h) or 1


__all__ = (
    "DEFAULT_BO_SIZE",
    "DEFAULT_ROUNDS",
    "DEFAULT_SAVE_PATH",
    "MAX_TIEBREAK_ROUNDS",
    "POINTS_PER_RESULT",
    "RANK_BONUSES",
    "MatchRecord",
    "PlayerStanding",
    "TournamentConfig",
    "TournamentState",
)
