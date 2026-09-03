# app/rag/text_normalizer.py
import re
import unicodedata


def normalize_embedding_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Embedding text must be a string")

    text = unicodedata.normalize("NFC", text)

    # Chuẩn hóa xuống dòng từ Windows/macOS.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    normalized_lines: list[str] = []

    for line in text.split("\n"):
        # Gộp space/tab liên tiếp, nhưng vẫn giữ cấu trúc từng dòng.
        line = re.sub(r"[ \t]+", " ", line).strip()

        if line:
            normalized_lines.append(line)

    normalized = "\n".join(normalized_lines)

    if not normalized:
        raise ValueError("Embedding text must not be empty")

    return normalized
