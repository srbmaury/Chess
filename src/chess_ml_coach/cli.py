import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def sync(username: str = "srbmaury") -> None:
    """Download and deduplicate Chess.com games."""


@app.command()
def analyze(username: str = "srbmaury") -> None:
    """Run resumable Stockfish analysis."""


@app.command()
def features(username: str = "srbmaury") -> None:
    """Create the ML-ready feature dataset."""


@app.command()
def train(username: str = "srbmaury") -> None:
    """Train the personalized mistake-risk model."""


@app.command()
def report(username: str = "srbmaury") -> None:
    """Generate the coaching report."""


if __name__ == "__main__":
    app()
