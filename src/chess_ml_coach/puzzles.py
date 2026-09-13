from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import chess
import pandas as pd

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


@dataclass(frozen=True)
class PuzzleSeed:
    puzzle_id: str
    game_id: str
    ply: int
    fen_before: str
    color: str
    game_label: str
    move_label: str
    your_move_san: str
    your_move_uci: str
    best_move_san: str
    best_move_uci: str
    cpl: int
    eval_loss_pawns: float
    quality: str
    opening: str
    eco: str
    game_phase: str
    source_url: str
    motif: str
    difficulty: int


def _text(value: object, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def _stable_puzzle_id(game_id: str, ply: int, best_move_uci: str) -> str:
    raw = f"{game_id}:{ply}:{best_move_uci}".encode("utf-8")
    return sha256(raw).hexdigest()[:24]


def _game_label(row: pd.Series) -> str:
    white = _text(row.get("white"))
    black = _text(row.get("black"))
    if white and black:
        return f"{white} vs {black}"
    return "Game"


def _move_label(row: pd.Series) -> str:
    move_number = int(row.get("fullmove_number") or ((int(row["ply"]) + 1) // 2))
    san = _text(row.get("san"), _text(row.get("uci"), "move"))
    return f"{move_number}.{san}" if row.get("color") == "white" else f"{move_number}...{san}"


def _newly_pinned_enemy_piece(
    before: chess.Board,
    after: chess.Board,
    enemy: chess.Color,
) -> bool:
    before_pinned = {
        square
        for square in before.piece_map()
        if before.color_at(square) == enemy
        and before.piece_type_at(square) != chess.KING
        and before.is_pinned(enemy, square)
    }
    after_pinned = {
        square
        for square in after.piece_map()
        if after.color_at(square) == enemy
        and after.piece_type_at(square) != chess.KING
        and after.is_pinned(enemy, square)
    }
    return bool(after_pinned - before_pinned)


def classify_motif(
    fen: str,
    best_move_uci: str,
    king_ring_attacks: int | None = None,
) -> str:
    board = chess.Board(fen)
    move = chess.Move.from_uci(best_move_uci)
    if move not in board.legal_moves:
        return "positional / calculation"

    before = board.copy(stack=False)
    mover = board.turn
    enemy = not mover
    moving_piece = board.piece_at(move.from_square)
    moving_value = PIECE_VALUES.get(moving_piece.piece_type, 0) if moving_piece else 0
    is_capture = board.is_capture(move)

    board.push(move)
    if board.is_checkmate():
        return "missed mate"
    if board.is_check():
        return "forcing check"

    attacked_targets = []
    for square in board.attacks(move.to_square):
        piece = board.piece_at(square)
        if piece is None or piece.color != enemy or piece.piece_type == chess.KING:
            continue
        if PIECE_VALUES.get(piece.piece_type, 0) > moving_value:
            attacked_targets.append(square)
    if len(attacked_targets) >= 2:
        return "fork"

    if _newly_pinned_enemy_piece(before, board, enemy):
        return "pin"
    if is_capture:
        return "winning capture"
    if king_ring_attacks is not None and int(king_ring_attacks) >= 3:
        return "king safety"
    return "positional / calculation"


def _difficulty(cpl: int, legal_move_count: int, motif: str) -> int:
    score = 1
    if cpl >= 200:
        score += 1
    if cpl >= 400:
        score += 1
    if legal_move_count >= 25:
        score += 1
    if motif == "positional / calculation":
        score += 1
    return max(1, min(5, score))


def _actual_move_delivers_mate(board: chess.Board, actual_uci: str) -> bool:
    try:
        move = chess.Move.from_uci(actual_uci)
    except ValueError:
        return False
    if move not in board.legal_moves:
        return False
    played = board.copy(stack=False)
    played.push(move)
    return played.is_checkmate()


def extract_puzzles(frame: pd.DataFrame) -> list[PuzzleSeed]:
    seeds: list[PuzzleSeed] = []
    for _, row in frame.iterrows():
        if _text(row.get("quality")).lower() not in {"mistake", "blunder"}:
            continue

        best_uci = _text(row.get("best_move_uci"))
        actual_uci = _text(row.get("uci"))
        fen = _text(row.get("fen_before"))
        if not best_uci or not actual_uci or not fen or best_uci == actual_uci:
            continue

        try:
            board = chess.Board(fen)
            best_move = chess.Move.from_uci(best_uci)
        except ValueError:
            continue
        if best_move not in board.legal_moves:
            continue
        if _actual_move_delivers_mate(board, actual_uci):
            continue

        cpl_value = pd.to_numeric(pd.Series([row.get("cpl")]), errors="coerce").iloc[0]
        if pd.isna(cpl_value):
            continue
        cpl = max(0, int(cpl_value))
        motif = classify_motif(
            fen,
            best_uci,
            int(row.get("king_ring_attacks", 0) or 0),
        )
        seeds.append(
            PuzzleSeed(
                puzzle_id=_stable_puzzle_id(_text(row.get("game_id")), int(row["ply"]), best_uci),
                game_id=_text(row.get("game_id")),
                ply=int(row["ply"]),
                fen_before=fen,
                color=_text(row.get("color"), "unknown"),
                game_label=_game_label(row),
                move_label=_move_label(row),
                your_move_san=_text(row.get("san"), actual_uci),
                your_move_uci=actual_uci,
                best_move_san=board.san(best_move),
                best_move_uci=best_uci,
                cpl=cpl,
                eval_loss_pawns=round(cpl / 100.0, 2),
                quality=_text(row.get("quality")).lower(),
                opening=_text(row.get("opening"), "Unknown opening"),
                eco=_text(row.get("eco"), "unknown"),
                game_phase=_text(row.get("game_phase"), "unknown"),
                source_url=_text(row.get("source_url")),
                motif=motif,
                difficulty=_difficulty(cpl, board.legal_moves.count(), motif),
            )
        )
    return seeds
