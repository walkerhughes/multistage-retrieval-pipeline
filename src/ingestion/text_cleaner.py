"""Text cleaning utilities for transcript processing."""


def clean_transcript_text(text: str) -> str:
    """
    Clean transcript text by removing unwanted characters.

    Removes:
    - Newline characters (\n)
    - Backslash characters (\)

    Args:
        text: Raw transcript text

    Returns:
        Cleaned text with newlines and backslashes removed
    """
    if not text:
        return text

    # Remove newline characters
    cleaned = text.replace("\n", " ")

    # Remove backslash characters
    cleaned = cleaned.replace("\\", "")

    # Collapse multiple spaces into single space
    cleaned = " ".join(cleaned.split())

    return cleaned
