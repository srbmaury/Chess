from pathlib import Path

from chess_ml_coach.pgn import parse_pgn_file

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pgn"
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"


def test_parser_marks_users_moves_for_white_and_black():
    games, moves = parse_pgn_file(FIXTURE, "srbmaury")
    assert len(games) == 2
    first = moves[moves.game_id == "https://www.chess.com/game/live/42"]
    second = moves[moves.game_id == "https://www.chess.com/game/live/43"]
    assert first.is_user_move.tolist() == [True, False, True, False]
    assert second.is_user_move.tolist() == [False, True, False, True]


def test_parser_captures_fen_and_clock_without_inventing_missing_values():
    _, moves = parse_pgn_file(FIXTURE, "srbmaury")
    first = moves[moves.game_id == "https://www.chess.com/game/live/42"].reset_index(drop=True)
    assert first.loc[0, "fen_before"] == START_FEN
    assert first.loc[0, "fen_after"] == AFTER_E4
    assert first.loc[0, "clock_seconds"] == 600.0
    assert first.loc[3, "clock_seconds"] != first.loc[3, "clock_seconds"]  # NaN


def test_game_metadata_is_normalized():
    games, _ = parse_pgn_file(FIXTURE, "srbmaury")
    white_game = games.iloc[0]
    assert white_game.game_id == "https://www.chess.com/game/live/42"
    assert white_game.white_rating == 1500
    assert white_game.black_rating == 1510
    assert white_game.eco == "C20"
    assert str(white_game.game_date.date()) == "2026-09-01"


def test_opening_name_falls_back_to_chesscom_eco_url(tmp_path: Path):
    pgn = tmp_path / "game.pgn"
    pgn.write_text(
        """[Event \"Live Chess\"]
[Site \"https://www.chess.com/game/live/99\"]
[Date \"2026.09.03\"]
[White \"srbmaury\"]
[Black \"opponent\"]
[Result \"1-0\"]
[ECO \"D00\"]
[ECOUrl \"https://www.chess.com/openings/Queens-Pawn-Opening\"]

1. d4 d5 1-0
""",
        encoding="utf-8",
    )

    games, _ = parse_pgn_file(pgn, "srbmaury")

    assert games.iloc[0].opening == "Queens Pawn Opening"
