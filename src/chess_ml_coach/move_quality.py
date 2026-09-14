from __future__ import annotations

from dataclasses import dataclass

import chess
import chess.engine

MATE_CP = 100_000
MATE_THRESHOLD_CP = 50_000


@dataclass(frozen=True)
class ScoreSnapshot:
    cp: int | None
    mate: int | None
    expected_score: float


@dataclass(frozen=True)
class MoveAssessment:
    label: str
    reason: str


def _expected_score(score: chess.engine.Score, *, ply: int = 30) -> float:
    try:
        wdl = score.wdl(model="sf16", ply=max(1, int(ply)))
    except (TypeError, ValueError):
        wdl = score.wdl()
    return (float(wdl.wins) + 0.5 * float(wdl.draws)) / 1000.0


def score_snapshot(
    score: chess.engine.PovScore,
    user_color: chess.Color,
    *,
    ply: int = 30,
) -> ScoreSnapshot:
    relative = score.pov(user_color)
    mate = relative.mate()
    cp = None if mate is not None else relative.score()
    return ScoreSnapshot(
        cp=int(cp) if cp is not None else None,
        mate=int(mate) if mate is not None else None,
        expected_score=_expected_score(relative, ply=ply),
    )


def snapshot_from_normalized_cp(value: int, *, ply: int = 30) -> ScoreSnapshot:
    value = int(value)
    if value >= MATE_THRESHOLD_CP:
        distance = max(0, MATE_CP - value)
        return ScoreSnapshot(cp=None, mate=distance, expected_score=1.0)
    if value <= -MATE_THRESHOLD_CP:
        distance = max(0, MATE_CP - abs(value))
        return ScoreSnapshot(cp=None, mate=-distance, expected_score=0.0)
    score = chess.engine.Cp(value)
    return ScoreSnapshot(cp=value, mate=None, expected_score=_expected_score(score, ply=ply))


def classify_move_quality(
    *,
    before: ScoreSnapshot,
    after: ScoreSnapshot,
    cpl: int,
    is_best_move: bool,
    brilliant_candidate: bool = False,
    delivered_mate: bool = False,
) -> MoveAssessment:
    cpl = max(0, int(cpl))
    expected_drop = max(0.0, before.expected_score - after.expected_score)

    if delivered_mate:
        return MoveAssessment("best", "delivers checkmate")

    if before.mate is not None and before.mate > 0 and not (after.mate is not None and after.mate >= 0):
        return MoveAssessment("miss", "forced mate was available")

    if after.mate is not None and after.mate < 0 and not (before.mate is not None and before.mate < 0):
        return MoveAssessment("blunder", "allows forced mate")

    if before.expected_score >= 0.90 and after.expected_score <= 0.65 and expected_drop >= 0.25:
        return MoveAssessment("miss", "decisive winning chance was missed")

    if (
        brilliant_candidate
        and cpl <= 30
        and expected_drop <= 0.03
        and after.expected_score >= 0.70
    ):
        return MoveAssessment("brilliant", "sound sacrifice and uniquely strong move")

    if is_best_move:
        return MoveAssessment("best", "engine top move")

    if cpl <= 10 and expected_drop <= 0.01:
        return MoveAssessment("best", "engine-equivalent move")

    if expected_drop >= 0.30:
        return MoveAssessment("blunder", "large drop in expected score")
    if expected_drop >= 0.15:
        return MoveAssessment("mistake", "significant drop in expected score")
    if expected_drop >= 0.05:
        return MoveAssessment("inaccuracy", "noticeable drop in expected score")

    if cpl <= 30 and expected_drop <= 0.02:
        return MoveAssessment("excellent", "near-best move")
    return MoveAssessment("good", "outcome essentially preserved")


def display_loss_pawns(cpl: int, reason: str = "") -> float | None:
    if abs(int(cpl)) >= MATE_THRESHOLD_CP or "forced mate" in reason.lower():
        return None
    return round(max(0, int(cpl)) / 100.0, 2)


def material_value(board: chess.Board, color: chess.Color) -> int:
    values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }
    return sum(len(board.pieces(piece_type, color)) * value for piece_type, value in values.items())


def sacrifices_material_after_reply(
    before_board: chess.Board,
    after_board: chess.Board,
    user_color: chess.Color,
    reply: chess.Move | None,
    *,
    minimum_material: int = 2,
) -> bool:
    if reply is None or reply not in after_board.legal_moves:
        return False
    replied = after_board.copy(stack=False)
    replied.push(reply)
    return material_value(before_board, user_color) - material_value(replied, user_color) >= minimum_material


def alternatives_show_uniqueness(
    scores: list[chess.engine.PovScore],
    user_color: chess.Color,
    *,
    ply: int = 30,
) -> bool:
    if len(scores) < 2:
        return False
    first = score_snapshot(scores[0], user_color, ply=ply)
    second = score_snapshot(scores[1], user_color, ply=ply)
    if first.mate is not None and first.mate >= 0 and not (second.mate is not None and second.mate >= 0):
        return True
    expected_gap = first.expected_score - second.expected_score
    if expected_gap >= 0.08:
        return True
    if first.cp is not None and second.cp is not None and first.cp - second.cp >= 100:
        return True
    return False
