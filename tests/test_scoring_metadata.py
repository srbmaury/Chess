import pandas as pd

from chess_ml_coach.train import _feature_columns


def test_scoring_version_is_not_used_as_a_model_feature():
    frame = pd.DataFrame({"signal": [1], "scoring_version": [2]})

    assert _feature_columns(frame) == ["signal"]
