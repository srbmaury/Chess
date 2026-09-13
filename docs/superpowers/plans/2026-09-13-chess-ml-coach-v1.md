# Chess ML Coach V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a private, CLI-first Python application that downloads `srbmaury`'s Chess.com games, evaluates the user's moves with Stockfish, creates an interpretable ML dataset, trains a personalized LightGBM mistake-risk model, and generates actionable coaching reports.

**Architecture:** Use a staged local pipeline with explicit artifacts between sync, parse, engine analysis, feature extraction, model training, and reporting. Keep network and Stockfish behind small adapters so all core logic is deterministic and unit-testable without live services.

**Tech Stack:** Python 3.11+, httpx, python-chess, pandas, pyarrow, numpy, LightGBM, scikit-learn, Typer, Rich, pytest, pytest-cov, ruff, GitHub Actions, external Stockfish binary.

**Spec:** `docs/superpowers/specs/2026-09-13-chess-ml-coach-design.md`

## Global Constraints

- Repository stays private.
- Default Chess.com username is `srbmaury`.
- Python version floor is 3.11.
- `data/`, `models/`, local caches, and environment files are gitignored.
- CI cannot depend on live Chess.com or a local Stockfish install.
- Training/test split is chronological at game granularity; no game may appear in both sets.
- Primary target is `significant_mistake = 1` for CPL >= 100.
- Default labels: good <50, inaccuracy 50-99, mistake 100-199, blunder >=200 CPL.
- Missing clocks remain null; never fabricate time data.
- Report groups below minimum sample size are suppressed.

---

### Task 1: Project foundation and CLI shell

**Files:**
- Create: `pyproject.toml`, `.gitignore`
- Create: `src/chess_ml_coach/__init__.py`, `config.py`, `cli.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Produces `MoveQualityThresholds`, `Settings`, `get_settings()`, Typer `app`.

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path
from typer.testing import CliRunner
from chess_ml_coach.cli import app
from chess_ml_coach.config import MoveQualityThresholds, Settings

runner = CliRunner()

def test_defaults(tmp_path: Path):
    s = Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models")
    assert s.username == "srbmaury"
    assert s.thresholds == MoveQualityThresholds(50, 100, 200)

def test_commands_exist():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ["sync", "analyze", "features", "train", "report"]:
        assert name in result.stdout
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_cli.py -v`
Expected: import/module failure.

- [ ] **Step 3: Implement minimum package**

`pyproject.toml` defines package `chess-ml-coach`, Python `>=3.11`, runtime dependencies `httpx`, `python-chess`, `pandas`, `pyarrow`, `numpy`, `lightgbm`, `scikit-learn`, `typer`, `rich`, plus dev dependencies `pytest`, `pytest-cov`, `ruff`. Script entrypoint: `chess-coach = "chess_ml_coach.cli:app"`.

```python
# config.py
from dataclasses import dataclass, field
from pathlib import Path
import os

@dataclass(frozen=True)
class MoveQualityThresholds:
    inaccuracy: int = 50
    mistake: int = 100
    blunder: int = 200

@dataclass(frozen=True)
class Settings:
    username: str = "srbmaury"
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    stockfish_path: str | None = None
    stockfish_depth: int = 14
    min_group_size: int = 10
    thresholds: MoveQualityThresholds = field(default_factory=MoveQualityThresholds)

def get_settings(username: str | None = None) -> Settings:
    return Settings(
        username=username or os.getenv("CHESS_COACH_USERNAME", "srbmaury"),
        data_dir=Path(os.getenv("CHESS_COACH_DATA_DIR", "data")),
        model_dir=Path(os.getenv("CHESS_COACH_MODEL_DIR", "models")),
        stockfish_path=os.getenv("STOCKFISH_PATH"),
        stockfish_depth=int(os.getenv("CHESS_COACH_STOCKFISH_DEPTH", "14")),
    )
```

`cli.py` exposes empty Typer command shells for all five stages.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_cli.py -v && ruff check src tests`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore src tests/test_cli.py
git commit -m "chore: scaffold chess coach CLI"
```

---

### Task 2: Chess.com synchronization

**Files:**
- Create: `src/chess_ml_coach/chesscom.py`
- Create: `tests/fixtures/archives.json`, `tests/fixtures/month.json`, `tests/test_chesscom.py`
- Modify: `src/chess_ml_coach/cli.py`

**Interfaces:**
- `ChessComClient.archive_urls(username) -> list[str]`
- `ChessComClient.games_for_archive(url) -> list[dict]`
- `canonical_game_id(game) -> str`
- `merge_games(existing, incoming) -> list[dict]`
- `sync_games(client, settings) -> SyncResult`

- [ ] **Step 1: Write failing pure tests**

```python
from chess_ml_coach.chesscom import canonical_game_id, merge_games

def test_url_is_primary_id():
    g = {"url": "https://www.chess.com/game/live/123", "pgn": "PGN"}
    assert canonical_game_id(g) == g["url"]

def test_merge_is_idempotent():
    a = [{"url": "https://www.chess.com/game/live/1", "pgn": "A"}]
    b = a + [{"url": "https://www.chess.com/game/live/2", "pgn": "B"}]
    merged = merge_games(a, b)
    assert len(merged) == 2
    assert merge_games(merged, b) == merged
```

Add `httpx.MockTransport` coverage for archive/month fixture parsing.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_chesscom.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement client and dedupe**

Use `https://api.chess.com/pub/player/{username}/games/archives`, transient retry for 429/5xx with exponential backoff, permanent failure for other 4xx. Prefer Chess.com game URL as stable ID; fallback to SHA-256 of PGN.

- [ ] **Step 4: Implement atomic sync artifacts**

Write:
- `data/raw/games.json`
- `data/raw/srbmaury_all_games.pgn`
- `data/raw/sync_manifest.json`

Persist after each monthly archive so partial progress survives interruption. Re-running identical input reports zero new games and creates no duplicates.

- [ ] **Step 5: Wire `sync`; verify GREEN**

Run: `pytest tests/test_chesscom.py tests/test_cli.py -v && ruff check src tests`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/chess_ml_coach/chesscom.py src/chess_ml_coach/cli.py tests/fixtures tests/test_chesscom.py
git commit -m "feat: sync Chess.com game archives"
```

---

### Task 3: PGN normalization

**Files:**
- Create: `src/chess_ml_coach/pgn.py`, `tests/fixtures/sample.pgn`, `tests/test_pgn.py`

**Interfaces:**
- `parse_pgn_file(path, username) -> tuple[pd.DataFrame, pd.DataFrame]`
- Move rows include `game_id`, `ply`, `fullmove_number`, `color`, `fen_before`, `fen_after`, `san`, `uci`, `is_user_move`, optional `clock_seconds`.

- [ ] **Step 1: Write failing ownership/FEN tests**

```python
from pathlib import Path
import chess
from chess_ml_coach.pgn import parse_pgn_file

def test_white_ownership_and_fens(tmp_path: Path):
    path = tmp_path / "g.pgn"
    path.write_text('[Site "https://www.chess.com/game/live/42"]\n[Date "2026.09.01"]\n[White "srbmaury"]\n[Black "opponent"]\n[Result "1-0"]\n\n1. e4 e5 2. Nf3 Nc6 1-0\n')
    games, moves = parse_pgn_file(path, "srbmaury")
    assert len(games) == 1
    assert moves.is_user_move.tolist() == [True, False, True, False]
    assert moves.iloc[0].fen_before == chess.Board().fen()
```

Add equivalent Black-side test.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_pgn.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement parser**

Use `chess.pgn.read_game()` sequentially. Capture board FEN before/after every mainline move and `node.clock()` when present. Use `Site` URL as game ID when available; otherwise hash normalized headers plus UCI movetext. Invalid games log a warning and do not abort the file.

- [ ] **Step 4: Implement deterministic parquet persistence**

Write sorted `games.parquet` and `moves.parquet`; validate required columns and raise `ArtifactSchemaError(path)` for corrupt/missing schema.

- [ ] **Step 5: Verify GREEN and commit**

Run: `pytest tests/test_pgn.py -v && ruff check src tests`

```bash
git add src/chess_ml_coach/pgn.py tests/fixtures/sample.pgn tests/test_pgn.py
git commit -m "feat: normalize PGN games and moves"
```

---

### Task 4: Stockfish analysis and CPL labels

**Files:**
- Create: `src/chess_ml_coach/engine.py`, `tests/test_engine.py`
- Modify: `src/chess_ml_coach/cli.py`

**Interfaces:**
- `EngineAdapter` protocol and `StockfishAdapter`
- `normalize_score(score, user_color) -> int`
- `quality_label(cpl, thresholds) -> str`
- `analyze_user_moves(...) -> pd.DataFrame`

- [ ] **Step 1: Write failing score/threshold tests**

```python
import chess, chess.engine
from chess_ml_coach.config import MoveQualityThresholds
from chess_ml_coach.engine import normalize_score, quality_label

def test_user_perspective():
    s = chess.engine.PovScore(chess.engine.Cp(80), chess.WHITE)
    assert normalize_score(s, chess.WHITE) == 80
    assert normalize_score(s, chess.BLACK) == -80

def test_thresholds():
    t = MoveQualityThresholds(50, 100, 200)
    assert quality_label(49, t) == "good"
    assert quality_label(50, t) == "inaccuracy"
    assert quality_label(100, t) == "mistake"
    assert quality_label(200, t) == "blunder"
```

Also assert mate scores map to bounded +/-100000.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_engine.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement adapter and score logic**

Stockfish uses `chess.engine.SimpleEngine.popen_uci(path)` and depth limit from settings. CPL compares best pre-move evaluation with played-move resulting evaluation from the user's perspective and clamps negative values to zero.

- [ ] **Step 4: Add fake-engine resumability test**

Use a deterministic fake adapter. Run analysis twice and assert `(game_id, ply, engine_config_hash)` rows are skipped on the second pass. Missing Stockfish path must fail before any output file is created.

- [ ] **Step 5: Wire `analyze`; verify and commit**

Run: `pytest tests/test_engine.py tests/test_cli.py -v && ruff check src tests`

```bash
git add src/chess_ml_coach/engine.py src/chess_ml_coach/cli.py tests/test_engine.py
git commit -m "feat: add resumable Stockfish analysis"
```

---

### Task 5: Interpretable features

**Files:**
- Create: `src/chess_ml_coach/features.py`, `tests/test_features.py`
- Modify: `src/chess_ml_coach/cli.py`

**Interfaces:**
- `extract_position_features(fen) -> dict`
- `build_feature_dataset(games, moves, analysis, thresholds) -> pd.DataFrame`

- [ ] **Step 1: Write failing starting-position test**

```python
import chess
from chess_ml_coach.features import extract_position_features

def test_start_position_features():
    f = extract_position_features(chess.STARTING_FEN)
    assert f["legal_move_count"] == 20
    assert f["total_non_king_material"] == 78
    assert f["white_queen_present"] == 1
    assert f["black_queen_present"] == 1
    assert f["in_check"] == 0
```

Add crafted doubled/isolated pawn assertions.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_features.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement pure board features**

Material values: P=1, N=3, B=3, R=5, Q=9. Include legal moves, material balance, total material, castling rights, queen presence, check state, pawn islands, doubled/isolated pawns, basic king-safety proxies.

Phase rule: opening <= move 12; endgame if no queens and total non-pawn non-king material <=16; else middlegame. Time-control categories: bullet <180s, blitz 180-599, rapid 600-3599, classical >=3600, else unknown.

- [ ] **Step 4: Join dataset and target test**

Only user moves survive. Join engine rows exactly on `(game_id, ply)`. Assert CPL 99 -> target 0 and CPL 100 -> target 1. Retain raw CPL/quality for reporting only.

- [ ] **Step 5: Wire `features`; verify and commit**

Run: `pytest tests/test_features.py -v && ruff check src tests`

```bash
git add src/chess_ml_coach/features.py src/chess_ml_coach/cli.py tests/test_features.py
git commit -m "feat: build interpretable chess features"
```

---

### Task 6: Leakage-safe LightGBM training

**Files:**
- Create: `src/chess_ml_coach/train.py`, `tests/test_train.py`
- Modify: `src/chess_ml_coach/cli.py`, `pyproject.toml`

**Interfaces:**
- `chronological_game_split(df, test_fraction=0.2)`
- `train_model(df, model_dir) -> TrainingResult`
- Save `mistake_model.joblib` + metadata JSON.

- [ ] **Step 1: Write failing split/guard tests**

```python
import pandas as pd, pytest
from chess_ml_coach.train import TrainingDataError, chronological_game_split

def test_no_game_leakage():
    df = pd.DataFrame({
        "game_id": ["g1","g1","g2","g2","g3","g3","g4","g4","g5","g5"],
        "game_date": pd.to_datetime(["2026-01-01"]*2+["2026-02-01"]*2+["2026-03-01"]*2+["2026-04-01"]*2+["2026-05-01"]*2),
        "significant_mistake": [0,1,0,0,1,0,1,1,0,1],
    })
    train, test = chronological_game_split(df, 0.4)
    assert set(train.game_id).isdisjoint(set(test.game_id))
    assert train.game_date.max() <= test.game_date.min()
```

Add single-class rejection.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_train.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement preprocessing/model**

Newest complete games form test partition. Use `ColumnTransformer` + `OneHotEncoder(handle_unknown="ignore")` for categoricals and numeric passthrough. Model: `LGBMClassifier(random_state=42, n_estimators=300, learning_rate=0.05, num_leaves=31)`. Persist full sklearn pipeline using `joblib` so preprocessing is reproducible.

- [ ] **Step 4: Metrics and metadata**

Report class balance, ROC-AUC, PR-AUC, log loss, Brier score when defined. Metadata records feature columns, thresholds, chronological cutoff, dataset hash/version, metrics. Reject too-small or one-class data with actionable `TrainingDataError`.

- [ ] **Step 5: Verify and commit**

Run: `pytest tests/test_train.py -v && ruff check src tests`

```bash
git add pyproject.toml src/chess_ml_coach/train.py src/chess_ml_coach/cli.py tests/test_train.py
git commit -m "feat: train personalized mistake model"
```

---

### Task 7: Coaching report

**Files:**
- Create: `src/chess_ml_coach/report.py`, `tests/test_report.py`
- Modify: `src/chess_ml_coach/cli.py`

**Interfaces:**
- `build_coaching_report(df, min_group_size=10) -> CoachingReport`
- `render_markdown(report) -> str`

- [ ] **Step 1: Write failing minimum-sample test**

```python
import pandas as pd
from chess_ml_coach.report import build_coaching_report

def test_small_opening_groups_suppressed():
    df = pd.DataFrame({
        "eco": ["C20"]*12 + ["B01"]*3,
        "significant_mistake": [0,1]*6 + [1,1,1],
        "quality": ["good","mistake"]*6 + ["blunder"]*3,
        "cpl": [10,120]*6 + [250]*3,
        "game_id": [f"g{i//3}" for i in range(15)],
        "ply": list(range(15)),
        "fen_before": ["fen"]*15,
        "color": ["white"]*15,
        "game_phase": ["middlegame"]*15,
        "time_control_category": ["rapid"]*15,
    })
    report = build_coaching_report(df, 10)
    assert "C20" in report.by_opening.index
    assert "B01" not in report.by_opening.index
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_report.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement guarded aggregations**

For each grouping compute `samples`, `mistake_rate`, `blunder_rate`, `mean_cpl`, then filter below minimum sample size. Rank contexts by `(mistake_rate - overall_mistake_rate) * log1p(samples)`. Candidate positions: top 20 CPL rows, maximum two per game.

- [ ] **Step 4: Markdown rendering test**

Assert sections: Overall, By color, By phase, Openings, Time controls, Recurring weakness contexts, Candidate training positions. Include `Feature importance is associative, not causal.` whenever model importance is shown.

- [ ] **Step 5: Wire `report`; verify and commit**

Run: `pytest tests/test_report.py -v && ruff check src tests`

```bash
git add src/chess_ml_coach/report.py src/chess_ml_coach/cli.py tests/test_report.py
git commit -m "feat: generate personalized coaching report"
```

---

### Task 8: End-to-end orchestration, README, CI, privacy verification

**Files:**
- Modify: `src/chess_ml_coach/cli.py`, `tests/test_cli.py`
- Create: `README.md`, `.github/workflows/test.yml`

- [ ] **Step 1: Add CLI smoke tests with monkeypatched stage functions**

Each command must use `srbmaury` by default, accept `--username`, report output artifact paths/counts, and return non-zero with a clear prerequisite message. `analyze` without configured Stockfish must explicitly mention `STOCKFISH_PATH`.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL until orchestration is complete.

- [ ] **Step 3: Complete orchestration**

Each command creates only its own output directory, validates required upstream artifacts, and calls exactly one stage service. Do not silently rebuild unrelated prior stages.

- [ ] **Step 4: Write README**

Document purpose/privacy, Python 3.11 setup, Stockfish + `STOCKFISH_PATH`, `pip install -e '.[dev]'`, exact five-command workflow, artifact layout, CPL labels/target, why LightGBM predicts personal mistake risk rather than best chess moves, and test/lint commands.

- [ ] **Step 5: Add GitHub Actions**

```yaml
name: test
on:
  push:
  pull_request:
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - run: python -m pip install --upgrade pip
      - run: pip install -e '.[dev]'
      - run: ruff check src tests
      - run: pytest --cov=chess_ml_coach --cov-report=term-missing
```

- [ ] **Step 6: Full verification**

```bash
ruff check src tests
pytest --cov=chess_ml_coach --cov-report=term-missing
python -m chess_ml_coach.cli --help
git check-ignore data/raw/example.pgn data/processed/moves.parquet data/engine/analysis.parquet models/mistake_model.joblib .env
```

Expected: all checks PASS and every personal/local artifact path is ignored.

- [ ] **Step 7: Commit**

```bash
git add README.md .github/workflows/test.yml src/chess_ml_coach/cli.py tests/test_cli.py
git commit -m "ci: verify reproducible chess coach pipeline"
```

## Final Verification Gate

Before claiming V1 complete:
1. `ruff check src tests` exits 0.
2. Full pytest suite exits 0.
3. No personal PGN/parquet/model/environment artifact is tracked.
4. Repository visibility is still private.
5. GitHub Actions for the final commit passes.
6. Only then run the live `sync` for `srbmaury`; generated personal data remains untracked.
