from __future__ import annotations

from collections import Counter
from itertools import pairwise
from pathlib import Path

import chess
import pandas as pd

from .config import MoveQualityThresholds

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def _pawn_files(board: chess.Board, color: chess.Color) -> list[int]:
    return [chess.square_file(square) for square in board.pieces(chess.PAWN, color)]


def _pawn_islands(files: list[int]) -> int:
    unique = sorted(set(files))
    if not unique:
        return 0
    return 1 + sum(1 for left, right in pairwise(unique) if right != left + 1)


def _doubled_pawns(files: list[int]) -> int:
    return sum(max(0, count - 1) for count in Counter(files).values())


def _isolated_pawns(files: list[int]) -> int:
    occupied = set(files)
    return sum(
        1
        for file_index in files
        if file_index - 1 not in occupied and file_index + 1 not in occupied
    )


def _material(board: chess.Board, color: chess.Color) -> int:
    return sum(
        len(board.pieces(piece_type, color)) * value
        for piece_type, value in PIECE_VALUES.items()
    )


def _non_pawn_non_king_material(board: chess.Board) -> int:
    return sum(
        len(board.pieces(piece_type, color)) * PIECE_VALUES[piece_type]
        for color in (chess.WHITE, chess.BLACK)
        for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )


def _king_ring_attacks(board: chess.Board, color: chess.Color) -> int:
    king_square = board.king(color)
    if king_square is None:
        return 0
    ring = chess.SquareSet(chess.BB_KING_ATTACKS[king_square])
    enemy = not color
    return sum(1 for square in ring if board.is_attacked_by(enemy, square))


def _king_castled(board: chess.Board, color: chess.Color) -> int:
    king_square = board.king(color)
    targets = {chess.G1, chess.C1} if color == chess.WHITE else {chess.G8, chess.C8}
    return int(king_square in targets)


def extract_position_features(fen: str) -> dict[str, int]:
    board = chess.Board(fen)
    white_files = _pawn_files(board, chess.WHITE)
    black_files = _pawn_files(board, chess.BLACK)
    white_material = _material(board, chess.WHITE)
    black_material = _material(board, chess.BLACK)
    return {
        "legal_move_count": board.legal_moves.count(),
        "in_check": int(board.is_check()),
        "white_material": white_material,
        "black_material": black_material,
        "material_balance_white": white_material - black_material,
        "total_non_king_material": white_material + black_material,
        "non_pawn_non_king_material": _non_pawn_non_king_material(board),
        "white_castling_rights": int(board.has_kingside_castling_rights(chess.WHITE))
        + int(board.has_queenside_castling_rights(chess.WHITE)),
        "black_castling_rights": int(board.has_kingside_castling_rights(chess.BLACK))
        + int(board.has_queenside_castling_rights(chess.BLACK)),
        "white_queen_present": int(bool(board.pieces(chess.QUEEN, chess.WHITE))),
        "black_queen_present": int(bool(board.pieces(chess.QUEEN, chess.BLACK))),
        "white_pawn_islands": _pawn_islands(white_files),
        "black_pawn_islands": _pawn_islands(black_files),
        "white_doubled_pawns": _doubled_pawns(white_files),
        "black_doubled_pawns": _doubled_pawns(black_files),
        "white_isolated_pawns": _isolated_pawns(white_files),
        "black_isolated_pawns": _isolated_pawns(black_files),
        "white_king_ring_attacks": _king_ring_attacks(board, chess.WHITE),
        "black_king_ring_attacks": _king_ring_attacks(board, chess.BLACK),
        "white_king_castled": _king_castled(board, chess.WHITE),
        "black_king_castled": _king_castled(board, chess.BLACK),
    }


def classify_phase(fen: str, fullmove_number: int) -> str:
    board = chess.Board(fen)
    if fullmove_number <= 12:
        return "opening"
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(
        board.pieces(chess.QUEEN, chess.BLACK)
    )
    if queens == 0 and _non_pawn_non_king_material(board) <= 16:
        return "endgame"
    return "middlegame"


def time_control_category(value: object) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    text = str(value).strip()
    if not text:
        return "unknown"
    if "/" in text:
        return "daily"
    try:
        seconds = float(text.split("+", 1)[0])
    except ValueError:
        return "unknown"
    if seconds < 180:
        return "bullet"
    if seconds < 600:
        return "blitz"
    if seconds < 3600:
        return "rapid"
    return "classical"


def build_feature_dataset(
    games: pd.DataFrame,
    moves: pd.DataFrame,
    analysis: pd.DataFrame,
    thresholds: MoveQualityThresholds,
) -> pd.DataFrame:
    user_moves = moves[moves["is_user_move"].fillna(False).astype(bool)].copy()
    if analysis.duplicated(["game_id", "ply"]).any():
        raise ValueError("Engine analysis contains duplicate game_id/ply rows")
    frame = user_moves.merge(
        analysis,
        on=["game_id", "ply"],
        how="inner",
        validate="one_to_one",
    )
    frame = frame.merge(games, on="game_id", how="left", validate="many_to_one")

    board_features = pd.DataFrame(
        [extract_position_features(fen) for fen in frame["fen_before"]],
        index=frame.index,
    )
    frame = pd.concat([frame, board_features], axis=1)
    frame["game_phase"] = [
        classify_phase(fen, int(move_number))
        for fen, move_number in zip(frame["fen_before"], frame["fullmove_number"])
    ]
    frame["time_control_category"] = frame["time_control"].map(time_control_category)
    frame["user_rating"] = frame.apply(
        lambda row: row.white_rating if row.color == "white" else row.black_rating,
        axis=1,
    )
    frame["opponent_rating"] = frame.apply(
        lambda row: row.black_rating if row.color == "white" else row.white_rating,
        axis=1,
    )
    frame["rating_difference"] = frame["user_rating"] - frame["opponent_rating"]
    frame["material_balance"] = frame.apply(
        lambda row: row.material_balance_white
        if row.color == "white"
        else -row.material_balance_white,
        axis=1,
    )
    frame["king_ring_attacks"] = frame.apply(
        lambda row: row.white_king_ring_attacks
        if row.color == "white"
        else row.black_king_ring_attacks,
        axis=1,
    )
    frame["own_castling_rights"] = frame.apply(
        lambda row: row.white_castling_rights
        if row.color == "white"
        else row.black_castling_rights,
        axis=1,
    )
    frame["own_doubled_pawns"] = frame.apply(
        lambda row: row.white_doubled_pawns
        if row.color == "white"
        else row.black_doubled_pawns,
        axis=1,
    )
    frame["own_isolated_pawns"] = frame.apply(
        lambda row: row.white_isolated_pawns
        if row.color == "white"
        else row.black_isolated_pawns,
        axis=1,
    )
    frame["own_pawn_islands"] = frame.apply(
        lambda row: row.white_pawn_islands
        if row.color == "white"
        else row.black_pawn_islands,
        axis=1,
    )
    frame["engine_eval_before_cp"] = frame["eval_before_cp"]
    frame["significant_mistake"] = frame["quality"].isin({"miss", "mistake", "blunder"}).astype(int)
    frame["mistake_cpl_threshold"] = thresholds.mistake
    frame["eco"] = frame["eco"].fillna("unknown")
    frame["opening"] = frame["opening"].fillna("unknown")
    return frame.sort_values(
        ["game_date", "game_id", "ply"],
        na_position="last",
    ).reset_index(drop=True)


def write_feature_dataset(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temp, index=False)
    temp.replace(path)
    return path
