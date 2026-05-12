from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ui.arena.tournament import (
    DEFAULT_BO_SIZE,
    DEFAULT_ROUNDS,
    DEFAULT_SAVE_PATH,
    MAX_TIEBREAK_ROUNDS,
    MatchRecord,
    TournamentConfig,
    TournamentState,
)


def _config(
    players: tuple[str, ...] = ("Felipe", "Julio", "Vicente"),
    tmp_path: Path | None = None,
    rounds: int = DEFAULT_ROUNDS,
    bo_size: int = DEFAULT_BO_SIZE,
) -> TournamentConfig:
    save = (tmp_path / "tournament.json") if tmp_path is not None else DEFAULT_SAVE_PATH
    return TournamentConfig(
        players=players,
        model_label="liga",
        mcts_sims=400,
        rounds=rounds,
        bo_size=bo_size,
        save_path=save,
    )


def _complete_all(state: TournamentState, *, by_player: dict[str, tuple[int, int]]) -> None:
    """Resuelve cada match con el marcador `(humano, ia)` provisto por jugador."""
    state.set_present(list(by_player))
    state.order = list(by_player)
    for player, (human, ai) in by_player.items():
        for match in state.matches_for(player):
            if match.status != "pending":
                continue
            state.record_result(match, human_pieces=human, ai_pieces=ai, halfmoves=40)


# ----- Construccion -----


def test_new_state_creates_rounds_times_bo_size_per_player() -> None:
    state = TournamentState.new(_config(), today="2026-05-12")
    assert state.tournament_id == "2026-05-12"
    # 3 jugadores * 2 rondas * 3 partidas = 18 matches.
    assert len(state.matches) == 3 * DEFAULT_ROUNDS * DEFAULT_BO_SIZE
    assert all(m.status == "pending" for m in state.matches)
    # Cada jugador tiene exactamente 6 matches regulares.
    for p in ("Felipe", "Julio", "Vicente"):
        assert len(state.matches_for(p)) == DEFAULT_ROUNDS * DEFAULT_BO_SIZE


def test_starter_follows_bo3_rule() -> None:
    """Dentro de cada Bo3: P1 sorteo, P2 invierte P1, P3 sorteo."""
    state = TournamentState.new(_config(), today="2026-05-12")
    for player in state.config.players:
        for round_idx in range(state.config.rounds):
            bo3 = [
                m for m in state.matches_for(player)
                if m.round_idx == round_idx
            ]
            bo3.sort(key=lambda m: m.match_idx)
            # P2 (match_idx=1) tiene starter opuesto a P1.
            assert bo3[1].starter != bo3[0].starter
            # P1, P3 son cada uno "human" o "ai" (no validamos correlacion).
            assert bo3[0].starter in {"human", "ai"}
            assert bo3[2].starter in {"human", "ai"}


def test_starter_is_reproducible_with_same_seed() -> None:
    a = TournamentState.new(_config(), today="2026-05-12")
    b = TournamentState.new(_config(), today="2026-05-12")
    starters_a = [(m.player, m.round_idx, m.match_idx, m.starter) for m in a.matches]
    starters_b = [(m.player, m.round_idx, m.match_idx, m.starter) for m in b.matches]
    assert starters_a == starters_b


# ----- Roster y orden -----


def test_set_present_marks_absentees_as_forfeit() -> None:
    state = TournamentState.new(_config())
    state.set_present(["Felipe", "Vicente"])
    julio_matches = state.matches_for("Julio")
    assert all(m.status == "forfeit" and m.result == "L" for m in julio_matches)
    felipe_matches = state.matches_for("Felipe")
    assert all(m.status == "pending" for m in felipe_matches)


def test_draw_order_only_shuffles_present_players() -> None:
    state = TournamentState.new(_config())
    state.set_present(["Felipe", "Vicente"])
    rng = random.Random(42)
    order = state.draw_order(rng)
    assert set(order) == {"Felipe", "Vicente"}
    assert state.order == order


# ----- Flujo de matches -----


def test_next_pending_walks_round_then_player_then_match() -> None:
    """Orden: ronda 0 completa (Vicente Bo3 -> Felipe Bo3 -> Julio Bo3), luego ronda 1."""
    state = TournamentState.new(_config())
    state.set_present(["Felipe", "Julio", "Vicente"])
    state.order = ["Vicente", "Felipe", "Julio"]
    # Primeros 3 matches: Vicente Bo3 de ronda 0.
    for expected_match_idx in range(3):
        m = state.next_pending_match()
        assert m is not None
        assert m.player == "Vicente"
        assert m.round_idx == 0
        assert m.match_idx == expected_match_idx
        state.record_result(m, human_pieces=24, ai_pieces=25, halfmoves=40)
    # Siguientes 3: Felipe Bo3 de ronda 0.
    for _ in range(3):
        m = state.next_pending_match()
        assert m is not None
        assert m.player == "Felipe"
        assert m.round_idx == 0
        state.record_result(m, human_pieces=24, ai_pieces=25, halfmoves=40)
    # Luego Julio Bo3 ronda 0.
    for _ in range(3):
        m = state.next_pending_match()
        assert m is not None
        assert m.player == "Julio"
        assert m.round_idx == 0
        state.record_result(m, human_pieces=24, ai_pieces=25, halfmoves=40)
    # Recien ahora arranca la ronda 1, otra vez Vicente primero.
    nxt = state.next_pending_match()
    assert nxt is not None
    assert nxt.player == "Vicente"
    assert nxt.round_idx == 1


def test_record_result_sets_w_l_d_correctly() -> None:
    state = TournamentState.new(_config())
    state.set_present(["Felipe"])
    match = state.matches_for("Felipe")[0]
    state.record_result(match, human_pieces=30, ai_pieces=15, halfmoves=42)
    assert match.result == "W"
    assert match.piece_diff == 15

    match2 = state.matches_for("Felipe")[1]
    state.record_result(match2, human_pieces=20, ai_pieces=20, halfmoves=42)
    assert match2.result == "D"

    match3 = state.matches_for("Felipe")[2]
    state.record_result(match3, human_pieces=10, ai_pieces=39, halfmoves=42)
    assert match3.result == "L"


def test_forfeit_remaining_for_marks_all_pendientes_as_l() -> None:
    state = TournamentState.new(_config())
    state.set_present(["Felipe", "Julio", "Vicente"])
    state.order = ["Felipe", "Julio", "Vicente"]
    # Julio juega solo una partida y se va.
    julio_matches = state.matches_for("Julio")
    state.record_result(julio_matches[0], human_pieces=24, ai_pieces=25, halfmoves=40)
    applied = state.forfeit_remaining_for("Julio")
    assert applied == 5
    julio_after = state.matches_for("Julio")
    # 1 completed L (24-25), 5 forfeit L.
    assert sum(1 for m in julio_after if m.status == "completed") == 1
    assert sum(1 for m in julio_after if m.status == "forfeit") == 5
    assert all(m.result == "L" for m in julio_after)


def test_finish_marks_pendientes_as_forfeit() -> None:
    state = TournamentState.new(_config())
    state.set_present(["Felipe", "Julio", "Vicente"])
    state.order = ["Felipe", "Julio", "Vicente"]
    # Una partida jugada, el resto queda pendiente.
    state.record_result(
        state.matches_for("Felipe")[0],
        human_pieces=30, ai_pieces=15, halfmoves=40,
    )
    state.finish()
    assert state.finished is True
    pendientes = [m for m in state.matches if m.status == "pending"]
    assert pendientes == []


# ----- Scoring agregado -----


def test_points_for_mixes_w_d_l_and_forfeit() -> None:
    state = TournamentState.new(_config(players=("A",)))
    state.set_present(["A"])
    matches = state.matches_for("A")
    # 2 W, 1 D, 1 L, 2 forfeit (resto se quedan pending sin contar).
    state.record_result(matches[0], human_pieces=30, ai_pieces=10, halfmoves=40)
    state.record_result(matches[1], human_pieces=25, ai_pieces=20, halfmoves=40)
    state.record_result(matches[2], human_pieces=20, ai_pieces=20, halfmoves=40)
    state.record_result(matches[3], human_pieces=15, ai_pieces=30, halfmoves=40)
    state.forfeit_remaining_for("A")  # marca matches[4] y matches[5] como L
    s = state.standing_for("A")
    assert s.wins == 2
    assert s.draws == 1
    assert s.losses == 3
    assert s.points == 2 * 1.0 + 1 * 0.5 + 3 * 0.0
    # piece_diff: +20, +5, 0, -15, 0, 0 = +10.
    assert s.piece_diff == 10
    assert s.played == 6


# ----- Ranking -----


def test_ranking_orders_by_points_then_wins_then_diff() -> None:
    state = TournamentState.new(_config(players=("A", "B", "C")))
    _complete_all(state, by_player={
        # A gana todas: 6 W, 6 puntos.
        "A": (30, 10),
        # B pierde todas: 6 L, 0 puntos.
        "B": (10, 30),
        # C empata todas: 6 D, 3 puntos.
        "C": (20, 20),
    })
    rank = state.ranking()
    assert [s.player for s in rank] == ["A", "C", "B"]
    assert rank[0].points == 6.0
    assert rank[1].points == 3.0
    assert rank[2].points == 0.0


def test_ranking_uses_wins_as_second_tiebreaker() -> None:
    state = TournamentState.new(_config(players=("A", "B"), rounds=2, bo_size=3))
    state.set_present(["A", "B"])
    state.order = ["A", "B"]
    # Ambos suman 3 puntos pero con perfiles distintos.
    # A: 3 W + 3 L = 3 pts, 3 wins, diff = -3 (3 wins por +20, 3 losses por -30 = -30+ ?). Forzamos:
    a_matches = state.matches_for("A")
    state.record_result(a_matches[0], human_pieces=25, ai_pieces=20, halfmoves=40)  # W +5
    state.record_result(a_matches[1], human_pieces=25, ai_pieces=20, halfmoves=40)  # W +5
    state.record_result(a_matches[2], human_pieces=25, ai_pieces=20, halfmoves=40)  # W +5
    state.record_result(a_matches[3], human_pieces=10, ai_pieces=20, halfmoves=40)  # L -10
    state.record_result(a_matches[4], human_pieces=10, ai_pieces=20, halfmoves=40)  # L -10
    state.record_result(a_matches[5], human_pieces=10, ai_pieces=20, halfmoves=40)  # L -10
    # A: 3 pts, 3 wins, diff -15.
    b_matches = state.matches_for("B")
    # B: 6 D = 3 pts, 0 wins, diff 0.
    for m in b_matches:
        state.record_result(m, human_pieces=22, ai_pieces=22, halfmoves=40)
    rank = state.ranking()
    # A va primero por mas victorias aunque tenga peor diff.
    assert [s.player for s in rank] == ["A", "B"]


def test_ranking_uses_piece_diff_as_third_tiebreaker() -> None:
    state = TournamentState.new(_config(players=("A", "B"), rounds=1, bo_size=2))
    state.set_present(["A", "B"])
    state.order = ["A", "B"]
    a = state.matches_for("A")
    b = state.matches_for("B")
    # Ambos: 1 W + 1 L = 1 pt, 1 win. Diferencia diferente.
    state.record_result(a[0], human_pieces=30, ai_pieces=15, halfmoves=40)  # +15
    state.record_result(a[1], human_pieces=15, ai_pieces=20, halfmoves=40)  # -5
    # A: diff = +10.
    state.record_result(b[0], human_pieces=22, ai_pieces=20, halfmoves=40)  # +2
    state.record_result(b[1], human_pieces=15, ai_pieces=20, halfmoves=40)  # -5
    # B: diff = -3.
    rank = state.ranking()
    assert [s.player for s in rank] == ["A", "B"]


def test_ranking_excludes_players_with_zero_played() -> None:
    state = TournamentState.new(_config())
    state.set_present(["Felipe", "Vicente"])  # Julio ausente -> sus matches estan en forfeit -> played > 0
    state.order = ["Felipe", "Vicente"]
    state.record_result(
        state.matches_for("Felipe")[0],
        human_pieces=30, ai_pieces=15, halfmoves=40,
    )
    # Vicente no jugo nada y esta presente: queda fuera del ranking todavia.
    rank = state.ranking()
    assert "Vicente" not in [s.player for s in rank]
    assert "Felipe" in [s.player for s in rank]
    # Julio si aparece (forfeit cuenta como played).
    assert "Julio" in [s.player for s in rank]


# ----- Tiebreak -----


def test_needs_tiebreak_detects_complete_tie_between_two() -> None:
    state = TournamentState.new(_config(players=("A", "B", "C"), rounds=1, bo_size=2))
    state.set_present(["A", "B", "C"])
    state.order = ["A", "B", "C"]
    # A y B empatan en todos los criterios; C queda lejos.
    for m in state.matches_for("A"):
        state.record_result(m, human_pieces=25, ai_pieces=20, halfmoves=40)  # 2 W, diff +10
    for m in state.matches_for("B"):
        state.record_result(m, human_pieces=25, ai_pieces=20, halfmoves=40)  # idem
    for m in state.matches_for("C"):
        state.record_result(m, human_pieces=10, ai_pieces=30, halfmoves=40)  # 2 L
    pair = state.needs_tiebreak()
    assert pair is not None
    assert set(pair) == {"A", "B"}


def test_needs_tiebreak_returns_none_when_no_tie() -> None:
    state = TournamentState.new(_config(players=("A", "B"), rounds=1, bo_size=2))
    _complete_all(state, by_player={"A": (30, 10), "B": (10, 30)})
    assert state.needs_tiebreak() is None


def test_tiebreak_pair_resolution_reorders_ranking() -> None:
    state = TournamentState.new(_config(players=("A", "B"), rounds=1, bo_size=2))
    _complete_all(state, by_player={"A": (25, 20), "B": (25, 20)})
    assert state.needs_tiebreak() == ("A", "B") or state.needs_tiebreak() == ("B", "A")
    pair = state.needs_tiebreak()
    assert pair is not None
    rec_a, rec_b = state.add_tiebreak_pair(pair[0], pair[1], rng=random.Random(0))
    # A pierde su relampago, B la gana.
    a_rec = rec_a if rec_a.player == "A" else rec_b
    b_rec = rec_b if rec_b.player == "B" else rec_a
    state.record_result(a_rec, human_pieces=10, ai_pieces=30, halfmoves=30)
    state.record_result(b_rec, human_pieces=30, ai_pieces=10, halfmoves=30)
    rank = state.ranking()
    # B termina arriba de A por ganar el desempate.
    assert [s.player for s in rank] == ["B", "A"]
    # Y ya no hay necesidad de otro tiebreak.
    assert state.needs_tiebreak() is None


def test_tiebreak_cap_prevents_infinite_loop() -> None:
    state = TournamentState.new(_config(players=("A", "B"), rounds=1, bo_size=2))
    _complete_all(state, by_player={"A": (25, 20), "B": (25, 20)})
    # Forzamos MAX_TIEBREAK_ROUNDS iteraciones todas terminadas en empate.
    for _ in range(MAX_TIEBREAK_ROUNDS):
        pair = state.needs_tiebreak()
        assert pair is not None
        a_rec, b_rec = state.add_tiebreak_pair(pair[0], pair[1], rng=random.Random(0))
        state.record_result(a_rec, human_pieces=20, ai_pieces=20, halfmoves=30)
        state.record_result(b_rec, human_pieces=20, ai_pieces=20, halfmoves=30)
    # Despues de N iteraciones, dejamos de pedir tiebreak.
    assert state.needs_tiebreak() is None


# ----- Persistencia -----


def test_persist_and_load_round_trip(tmp_path: Path) -> None:
    config = _config(tmp_path=tmp_path)
    state = TournamentState.new(config, today="2026-05-12")
    state.set_present(["Felipe", "Vicente"])
    state.draw_order(random.Random(7))
    first = state.matches_for("Felipe")[0]
    state.record_result(first, human_pieces=24, ai_pieces=25, halfmoves=38)
    state.persist()
    loaded = TournamentState.load(config.save_path)
    assert loaded is not None
    assert loaded.tournament_id == "2026-05-12"
    assert loaded.order == state.order
    # Los starters se preservan: clave para resume sin re-sortear.
    original = [(m.player, m.round_idx, m.match_idx, m.starter) for m in state.matches]
    reloaded = [(m.player, m.round_idx, m.match_idx, m.starter) for m in loaded.matches]
    assert original == reloaded


def test_load_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert TournamentState.load(tmp_path / "nope.json") is None


def test_load_returns_none_for_corrupt_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    assert TournamentState.load(bad) is None


def test_match_record_dict_round_trip() -> None:
    m = MatchRecord(
        player="Felipe",
        round_idx=1,
        match_idx=2,
        starter="ai",
        status="completed",
        result="L",
        human_pieces=20,
        ai_pieces=25,
        piece_diff=-5,
        halfmoves=38,
        played_at=1234.5,
    )
    payload = json.loads(json.dumps(m.to_dict()))
    assert MatchRecord.from_dict(payload) == m
