"""Transcript loader for markdown files."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evals.schemas.task import EvalDataset

logger = logging.getLogger(__name__)


@dataclass
class Transcript:
    """Loaded transcript document."""

    filename: str
    speaker: str
    title: str
    text: str
    file_path: str


class TranscriptLoader:
    """Load and parse markdown transcripts from the transcripts directory.

    Parses markdown files from the transcripts/ directory, extracting speaker
    and title metadata from filenames.

    Example:
        >>> loader = TranscriptLoader()
        >>> transcripts = loader.load_all()
        >>> for t in transcripts:
        ...     print(f"{t.speaker}: {t.title}")
    """

    def __init__(self, transcripts_dir: str | Path | None = None):
        """Initialize the loader.

        Args:
            transcripts_dir: Path to transcripts directory.
                           Defaults to PROJECT_ROOT/transcripts
        """
        if transcripts_dir is None:
            # Default to transcripts/ in project root
            self.transcripts_dir = self._get_project_root() / "transcripts"
        else:
            self.transcripts_dir = Path(transcripts_dir)

    @staticmethod
    def _get_project_root() -> Path:
        """Get the project root directory."""
        # Navigate up from evals/tasks/retrieval/loaders/transcript_loader.py
        return Path(__file__).parent.parent / "datasets"

    def load(self, filename: str) -> Transcript:
        """Load a single transcript by filename.

        Args:
            filename: Name of the markdown file (with or without .md extension)

        Returns:
            Transcript object with parsed content and metadata

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is empty or has invalid format
        """
        # Ensure .md extension
        if not filename.endswith(".md"):
            filename = f"{filename}.md"

        filepath = self.transcripts_dir / filename

        if not filepath.exists():
            available = [f.name for f in self.transcripts_dir.glob("*.md")]
            raise FileNotFoundError(
                f"Transcript not found: {filepath}\n"
                f"Available transcripts: {available}"
            )

        text = filepath.read_text(encoding="utf-8")

        if not text.strip():
            raise ValueError(f"Transcript file is empty: {filepath}")

        speaker, title = self._parse_filename(filename)

        return Transcript(
            filename=filename,
            speaker=speaker,
            title=title,
            text=text,
            file_path=str(filepath),
        )

    def load_all(self) -> list[Transcript]:
        """Load all markdown transcripts from the transcripts directory.

        Returns:
            List of Transcript objects, sorted alphabetically by filename

        Raises:
            FileNotFoundError: If the transcripts directory doesn't exist
        """
        if not self.transcripts_dir.exists():
            raise FileNotFoundError(
                f"Transcripts directory not found: {self.transcripts_dir}"
            )

        transcripts = []
        for filepath in sorted(self.transcripts_dir.glob("*.md")):
            try:
                transcripts.append(self.load(filepath.name))
            except ValueError as e:
                # Skip empty or invalid files but log the issue
                logger.warning("Skipping invalid transcript: %s", e)

        return transcripts

    def get_transcript_names(self) -> list[str]:
        """Get list of available transcript filenames.

        Returns:
            List of markdown filenames in the transcripts directory
        """
        if not self.transcripts_dir.exists():
            return []
        return sorted(f.name for f in self.transcripts_dir.glob("*.md"))

    @staticmethod
    def _parse_filename(filename: str) -> tuple[str, str]:
        """Extract speaker and title from transcript filename.

        Filenames follow the pattern: {firstname}-{lastname}-{title-words}.md
        The first two hyphen-separated words are the speaker name,
        and the remaining words form the title.

        Args:
            filename: The transcript filename

        Returns:
            Tuple of (speaker, title) with proper formatting

        Examples:
            >>> TranscriptLoader._parse_filename(
            ...     "ilya-sutskever-were-moving-from-the-age-of-scaling.md"
            ... )
            ('Ilya Sutskever', "We're Moving From The Age Of Scaling")
        """
        # Remove .md extension
        name = filename.replace(".md", "")

        # Split on hyphens
        parts = name.split("-")

        if len(parts) >= 3:
            # First two parts are speaker name, rest is title
            speaker_part = " ".join(parts[:2])
            title_part = " ".join(parts[2:])
        elif len(parts) == 2:
            # Could be "firstname-lastname" with no title
            speaker_part = " ".join(parts)
            title_part = ""
        else:
            # Fallback: use whole filename as title
            speaker_part = "Unknown"
            title_part = name

        # Clean up speaker: title case
        speaker = speaker_part.title().strip()
        if not speaker:
            speaker = "Unknown"

        # Clean up title: title case
        title = title_part.title().strip()
        if not title:
            title = "Untitled"

        # Fix common contractions that title() breaks
        title = title.replace("'S ", "'s ").replace("'T ", "'t ").replace("'Re ", "'re ")

        return speaker, title

    def search_text(self, query: str, case_sensitive: bool = False) -> list[tuple[Transcript, list[str]]]:
        """Search all transcripts for a query string.

        Useful for finding relevant sections when curating eval questions.

        Args:
            query: Text to search for
            case_sensitive: Whether to do case-sensitive search

        Returns:
            List of (transcript, matching_lines) tuples
        """
        results = []

        for transcript in self.load_all():
            lines = transcript.text.split("\n")
            matches = []

            for line in lines:
                check_line = line if case_sensitive else line.lower()
                check_query = query if case_sensitive else query.lower()

                if check_query in check_line:
                    matches.append(line.strip())

            if matches:
                results.append((transcript, matches))

        return results


def load_eval_dataset(filepath: str | Path | None = None) -> EvalDataset:
    """Load and validate the eval questions JSON file.

    Args:
        filepath: Path to JSON file. Defaults to evals/datasets/eval_questions.json

    Returns:
        EvalDataset object with validated examples

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValidationError: If JSON doesn't match the schema
    """
    # Import here to avoid circular imports at module load time
    from evals.schemas.task import EvalDataset

    if filepath is None:
        filepath = Path(__file__).parent.parent / "datasets" / "eval_questions.json"
    else:
        filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Eval dataset not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return EvalDataset(**data)
