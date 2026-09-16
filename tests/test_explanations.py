from datetime import UTC, datetime
from pathlib import Path

import chess
import chess.engine

from chess_ml_coach import explanations
from chess_ml_coach.config import Settings
from chess_ml_coach.puzzles import PuzzleSeed
from chess_ml_coach.training import TrainingStore

TACTICAL_FEN = "6k1/5ppp/8/7Q/8/8/6PP/6K1 w - - 0 1"


def _seed() -> PuzzleSeed:
    return PuzzleSeed(
        puzzle_id="p1",
        game_id="g1",
        ply=1,
        fen_before=TACTICAL_FEN,
        color="white",
        game_label="srbmaury vs opponent",
        move_label="1.Qh3",
        your_move_san="Qh3",
        your_move_uci="h5h3",
        best_move_san="Qxh7+",
        best_move_uci="h5h7",
        cpl=250,
        eval_loss_pawns=2.5,
        quality="blunder",
        opening="Test opening",
        eco="A00",
        game_phase="middlegame",
        source_url="https://www.chess.com/game/live/1",
        motif="forcing check",
        difficulty=4,
    )


def _puzzle(tmp_path: Path):
    db = tmp_path / "training.db"
    store = TrainingStore(db)
    store.upsert_puzzles([_seed()], now=datetime(2026, 1, 1, tzinfo=UTC))
    puzzle = store.get_puzzle("p1")
    assert puzzle is not None
    return puzzle, explanations.ExplanationStore(db)


def test_first_explanation_uses_engine_then_reuses_cache(tmp_path: Path):
    puzzle, cache = _puzzle(tmp_path)
    calls = []

    def analyse(board: chess.Board, depth: int) -> dict:
        calls.append((board.fen(), depth))
        return {
            "score": chess.engine.PovScore(chess.engine.Cp(450), chess.WHITE),
            "pv": [
                chess.Move.from_uci("h5h7"),
                chess.Move.from_uci("g8f8"),
                chess.Move.from_uci("h7h8"),
            ],
        }

    settings = Settings(stockfish_depth=14)
    service = explanations.PuzzleExplanationService(settings, cache, analyse=analyse)

    first = service.explain(puzzle)
    second = service.explain(puzzle)

    assert len(calls) == 1
    assert first["cached"] is False
    assert second["cached"] is True
    assert first["engine_grounded"] is True
    assert first["depth"] == 14
    assert first["best_line"][0] == "Qxh7+"
    assert "check" in str(first["why"]).lower() or "king" in str(first["why"]).lower()
    assert "2.50" in str(first["why_your_move_was_worse"])


def test_engine_failure_returns_uncached_board_fallback(tmp_path: Path):
    puzzle, cache = _puzzle(tmp_path)
    calls = 0

    def fail(_board: chess.Board, _depth: int) -> dict:
        nonlocal calls
        calls += 1
        raise RuntimeError("stockfish unavailable")

    service = explanations.PuzzleExplanationService(
        Settings(stockfish_depth=14), cache, analyse=fail
    )

    first = service.explain(puzzle)
    second = service.explain(puzzle)

    assert calls == 2
    assert first["engine_grounded"] is False
    assert first["cached"] is False
    assert first["best_line"] == ["Qxh7+"]
    assert "Stockfish" in str(first["why"])
    assert second["cached"] is False


def test_engine_disagreement_is_not_cached_or_used_as_the_stored_move_explanation(
    tmp_path: Path,
):
    puzzle, cache = _puzzle(tmp_path)
    calls = 0

    def analyse(_board: chess.Board, _depth: int) -> dict:
        nonlocal calls
        calls += 1
        return {
            "score": chess.engine.PovScore(chess.engine.Cp(80), chess.WHITE),
            "pv": [chess.Move.from_uci("h5h3")],
        }

    service = explanations.PuzzleExplanationService(
        Settings(stockfish_depth=14), cache, analyse=analyse
    )

    first = service.explain(puzzle)
    second = service.explain(puzzle)

    assert calls == 2
    assert first["engine_grounded"] is False
    assert first["cached"] is False
    assert first["best_line"] == ["Qxh7+"]
    assert "refresh" in str(first["why"]).lower()
    assert "Qh3" in str(first["why"])
    assert second["cached"] is False


def test_hanging_piece_note_names_the_free_capture():
    # Real position from a srbmaury game: white's queen on h6 can walk into
    # ...Qxh5 along the open 5th rank - a queen hanging for free.
    board = chess.Board("r3kr2/pp2b2p/5p1Q/3q4/8/8/PPP2PPP/RNB1KB1R w KQq - 1 13")
    move = chess.Move.from_uci("h6h5")

    note = explanations._hanging_piece_note(board, move)

    assert note is not None
    assert "queen" in note
    assert "h5" in note
    assert "Qxh5" in note


def test_hanging_piece_note_is_none_when_the_piece_is_defended():
    board = chess.Board(TACTICAL_FEN)
    move = chess.Move.from_uci("h5h3")

    assert explanations._hanging_piece_note(board, move) is None


def test_explanation_names_the_hanging_piece_when_the_move_loses_it_for_free(
    tmp_path: Path,
):
    puzzle = PuzzleSeed(
        puzzle_id="p2",
        game_id="g2",
        ply=13,
        fen_before="r3kr2/pp2b2p/5p1Q/3q4/8/8/PPP2PPP/RNB1KB1R w KQq - 1 13",
        color="white",
        game_label="srbmaury vs opponent",
        move_label="13.Qh5+",
        your_move_san="Qh5+",
        your_move_uci="h6h5",
        best_move_san="Qh3",
        best_move_uci="h6h3",
        cpl=1315,
        eval_loss_pawns=13.15,
        quality="miss",
        opening="Test opening",
        eco="A00",
        game_phase="middlegame",
        source_url="https://www.chess.com/game/live/1",
        motif="positional / calculation",
        difficulty=4,
    )
    db = tmp_path / "training.db"
    store = TrainingStore(db)
    store.upsert_puzzles([puzzle], now=datetime(2026, 1, 1, tzinfo=UTC))
    stored = store.get_puzzle("p2")
    assert stored is not None
    cache = explanations.ExplanationStore(db)

    def analyse(_board: chess.Board, _depth: int) -> dict:
        return {
            "score": chess.engine.PovScore(chess.engine.Cp(900), chess.WHITE),
            "pv": [chess.Move.from_uci("h6h3")],
        }

    service = explanations.PuzzleExplanationService(
        Settings(stockfish_depth=14), cache, analyse=analyse
    )

    result = service.explain(stored)

    why_worse = str(result["why_your_move_was_worse"])
    assert "Qxh5" in why_worse
    assert "undefended" in why_worse
    assert "13.15" in why_worse
