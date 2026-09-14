import pandas as pd

from chess_ml_coach.train import _feature_columns


def test_scoring_version_is_not_used_as_a_model_feature():
    frame = pd.DataFrame({"signal": [1], "scoring_version": [3]})

    assert _feature_columns(frame) == ["signal"]


def test_post_move_quality_metadata_is_not_used_as_model_features():
    frame = pd.DataFrame(
        {
            "signal": [1],
            "mate_after": [-3],
            "expected_score_after": [0.0],
            "quality_reason": ["allows forced mate"],
        }
    )

    assert _feature_columns(frame) == ["signal"]
