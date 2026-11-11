import io
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def open_puzzle_csv(path: Path):
    """Open the Lichess puzzle CSV as text, transparently handling .zst."""
    if path.suffix == ".zst":
        import zstandard

        with open(path, "rb") as raw:
            reader = zstandard.ZstdDecompressor().stream_reader(raw)
            yield io.TextIOWrapper(reader, encoding="utf-8", newline="")
    else:
        with open(path, encoding="utf-8", newline="") as fh:
            yield fh
