from __future__ import annotations

from pathlib import Path


def load_tokenmonster_vocab(vocab_ref: str):
    try:
        import tokenmonster
    except ImportError as exc:
        raise RuntimeError("tokenmonster is required") from exc

    path = Path(vocab_ref).expanduser()
    if path.suffix.lower() in {".yaml", ".yml"} and path.is_file():
        return tokenmonster.new(path.read_bytes())
    return tokenmonster.load(vocab_ref)
