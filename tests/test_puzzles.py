import pandas as pd

from chess_ml_coach.puzzles import classify_motif, extract_puzzles

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
FOOLS_MATE_BEFORE = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq g3 0 2"


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "g1",
                "ply": 1,
                "fullmove_number": 1,
                "color": "white",
                "fen_before": START_FEN,
                "fen_after": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
                "san": "e4",
                "uci": "e2e4",
                "best_move_uci": "d2d4",
                "cpl": 250,
                "quality": "blunder",
                "quality_reason": "large drop in expected score",
                "white": "srbmaury",
                "black": "opponent",
                "opening": "King's Pawn Opening",
                "eco": "C20",
                "game_phase": "opening",
                "source_url": "https://www.chess.com/game/live/1",
                "king_ring_attacks": 0,
            }
        ]
    )


def test_extracts_stable_human_readable_puzzle_seed():
    first = extract_puzzles(_frame())
    second = extract_puzzles(_frame())

    assert len(first) == 1
    puzzle = first[0]
    assert puzzle.puzzle_id == second[0].puzzle_id
    assert puzzle.game_label == "srbmaury vs opponent"
    assert puzzle.move_label == "1.e4"
    assert puzzle.your_move_san == "e4"
    assert puzzle.best_move_san == "d4"
    assert puzzle.best_move_uci == "d2d4"
    assert puzzle.eval_loss_pawns == 2.5
    assert puzzle.quality_reason == "large drop in expected score"
    assert puzzle.difficulty in {1, 2, 3, 4, 5}


def test_miss_is_training_worthy_and_forced_mate_loss_is_not_shown_as_pawns():
    frame = _frame()
    frame.loc[0, "quality"] = "miss"
    frame.loc[0, "quality_reason"] = "forced mate was available"
    frame.loc[0, "cpl"] = 99_808

    puzzles = extract_puzzles(frame)

    assert len(puzzles) == 1
    assert puzzles[0].quality == "miss"
    assert puzzles[0].quality_reason == "forced mate was available"
    assert puzzles[0].eval_loss_pawns is None


def test_same_move_false_blunder_is_excluded():
    frame = _frame()
    frame.loc[0, "best_move_uci"] = "e2e4"

    assert extract_puzzles(frame) == []


def test_player_move_that_delivers_mate_is_excluded_even_if_best_move_is_wrong():
    frame = _frame()
    frame.loc[0, "color"] = "black"
    frame.loc[0, "fen_before"] = FOOLS_MATE_BEFORE
    frame.loc[0, "san"] = "Qh4#"
    frame.loc[0, "uci"] = "d8h4"
    frame.loc[0, "best_move_uci"] = "a7a6"

    assert extract_puzzles(frame) == []


def test_non_mistakes_and_missing_best_move_are_excluded():
    frame = pd.concat([_frame(), _frame()], ignore_index=True)
    frame.loc[0, "quality"] = "good"
    frame.loc[1, "best_move_uci"] = None

    assert extract_puzzles(frame) == []


def test_classifies_missed_mate_before_other_motifs():
    assert classify_motif(FOOLS_MATE_BEFORE, "d8h4") == "missed mate"


def test_classifies_forcing_check():
    fen = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
    assert classify_motif(fen, "d1h5") == "forcing check"


def test_classifies_knight_fork_of_two_higher_value_pieces():
    fen = "7k/8/8/8/1r3q2/2N5/8/K7 w - - 0 1"
    assert classify_motif(fen, "c3d5") == "fork"


def test_classifies_winning_capture():
    fen = "8/7k/8/8/8/8/8/qR5K w - - 0 1"
    assert classify_motif(fen, "b1a1") == "winning capture"
