from pathlib import Path


def read_file(path: str) -> str:
    """
    Read text content from a local file.

    This is a real file operation. SentinelAI will later
    record the resulting content as a DataObject.
    """

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    return file_path.read_text(encoding="utf-8")


def summarize(text: str, max_sentences: int = 3) -> str:
    """
    Create a simple extractive summary from text.

    This is intentionally deterministic so that the transformation
    is reproducible during experiments.
    """

    if not text.strip():
        return ""

    sentences = [
        sentence.strip()
        for sentence in text.replace("!", ".").replace("?", ".").split(".")
        if sentence.strip()
    ]

    return ". ".join(sentences[:max_sentences]) + (
        "." if sentences[:max_sentences] else ""
    )


def paraphrase(text: str) -> str:
    """
    Create a deterministic paraphrased version of text.

    This is a baseline transformation that will later allow
    SentinelAI to track information across transformed objects.
    """

    if not text.strip():
        return ""

    replacements = {
        "is": "remains",
        "are": "remain",
        "uses": "utilizes",
        "shows": "demonstrates",
        "helps": "assists",
        "detects": "identifies",
    }

    result = text

    for original, replacement in replacements.items():
        result = result.replace(
            f" {original} ",
            f" {replacement} "
        )

    return result


def calculate(expression: str) -> float:
    """
    Evaluate a basic arithmetic expression.

    Only numeric arithmetic is supported.
    """

    allowed_characters = set("0123456789+-*/(). ")

    if not expression.strip():
        raise ValueError("Expression cannot be empty.")

    if not set(expression).issubset(allowed_characters):
        raise ValueError("Expression contains unsupported characters.")

    try:
        result = eval(expression, {"__builtins__": {}}, {})
    except (SyntaxError, ZeroDivisionError, TypeError) as exc:
        raise ValueError(f"Invalid arithmetic expression: {expression}") from exc

    if not isinstance(result, (int, float)):
        raise ValueError("Expression did not produce a numeric result.")

    return float(result)