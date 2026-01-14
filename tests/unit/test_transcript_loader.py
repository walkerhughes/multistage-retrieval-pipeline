"""Unit tests for TranscriptLoader."""

import tempfile
from pathlib import Path

import pytest

from evals.tasks.retrieval.loaders.transcript_loader import (
    Transcript,
    TranscriptLoader,
    load_eval_dataset,
)
from evals.schemas.task import EvalDataset


class TestTranscriptLoader:
    """Unit tests for TranscriptLoader class."""

    @pytest.fixture
    def loader(self) -> TranscriptLoader:
        """Create a TranscriptLoader instance using the real transcripts directory."""
        return TranscriptLoader()

    @pytest.fixture
    def temp_loader(self, tmp_path: Path) -> TranscriptLoader:
        """Create a TranscriptLoader with a temporary directory."""
        return TranscriptLoader(transcripts_dir=tmp_path)

    def test_load_single_transcript(self, loader: TranscriptLoader):
        """Test loading a single existing transcript."""
        # Load the Andrej Karpathy transcript
        transcript = loader.load(
            "andrej-karpathy-were-summoning-ghosts-not-building-animals.md"
        )

        assert isinstance(transcript, Transcript)
        assert transcript.filename.endswith(".md")
        assert len(transcript.text) > 1000  # Transcript should be substantial
        assert transcript.speaker != ""
        assert transcript.title != ""

    def test_load_transcript_adds_extension(self, loader: TranscriptLoader):
        """Test that .md extension is added if missing."""
        transcript = loader.load(
            "andrej-karpathy-were-summoning-ghosts-not-building-animals"
        )
        assert transcript.filename.endswith(".md")

    def test_load_nonexistent_file(self, loader: TranscriptLoader):
        """Test error handling for missing files."""
        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load("nonexistent-transcript.md")

        assert "Transcript not found" in str(exc_info.value)
        assert "Available transcripts" in str(exc_info.value)

    def test_load_empty_file(self, temp_loader: TranscriptLoader, tmp_path: Path):
        """Test error handling for empty files."""
        # Create an empty file
        empty_file = tmp_path / "empty-transcript.md"
        empty_file.write_text("")

        with pytest.raises(ValueError) as exc_info:
            temp_loader.load("empty-transcript.md")

        assert "empty" in str(exc_info.value).lower()

    def test_load_all_transcripts(self, loader: TranscriptLoader):
        """Test loading all transcripts from the directory."""
        transcripts = loader.load_all()

        assert len(transcripts) >= 3  # We have 3 transcripts
        assert all(isinstance(t, Transcript) for t in transcripts)
        assert all(t.text for t in transcripts)

    def test_load_all_sorted(self, loader: TranscriptLoader):
        """Test that load_all returns transcripts sorted by filename."""
        transcripts = loader.load_all()
        filenames = [t.filename for t in transcripts]
        assert filenames == sorted(filenames)

    def test_get_transcript_names(self, loader: TranscriptLoader):
        """Test getting list of transcript filenames."""
        names = loader.get_transcript_names()

        assert len(names) >= 3
        assert all(name.endswith(".md") for name in names)
        assert names == sorted(names)

    def test_get_transcript_names_empty_dir(self, temp_loader: TranscriptLoader):
        """Test getting names from empty directory."""
        names = temp_loader.get_transcript_names()
        assert names == []


class TestFilenameParser:
    """Unit tests for filename parsing logic."""

    def test_parse_filename_standard(self):
        """Test parsing standard filename format (firstname-lastname-title-words)."""
        speaker, title = TranscriptLoader._parse_filename(
            "ilya-sutskever-were-moving-from-the-age-of-scaling.md"
        )

        assert speaker == "Ilya Sutskever"
        assert "Moving" in title or "Age" in title

    def test_parse_filename_with_long_title(self):
        """Test parsing filename with many title words."""
        speaker, title = TranscriptLoader._parse_filename(
            "andrej-karpathy-were-summoning-ghosts-not-building-animals.md"
        )

        assert speaker == "Andrej Karpathy"
        assert "Summoning" in title or "Ghosts" in title

    def test_parse_filename_three_parts(self):
        """Test parsing filename with minimum three parts."""
        speaker, title = TranscriptLoader._parse_filename("john-doe-hello.md")

        assert speaker == "John Doe"
        assert title == "Hello"

    def test_parse_filename_preserves_contractions(self):
        """Test that common contractions are preserved."""
        speaker, title = TranscriptLoader._parse_filename(
            "john-doe-we're-building-it's-great.md"
        )

        # Contractions should be lowercase after title case conversion
        assert "'re" in title.lower() or "'s" in title.lower()

    def test_parse_filename_only_name(self):
        """Test handling of filename with only speaker name (two parts)."""
        speaker, title = TranscriptLoader._parse_filename("speaker-name.md")

        assert speaker == "Speaker Name"
        assert title == "Untitled"

    def test_parse_filename_single_word(self):
        """Test handling of single word filename."""
        speaker, title = TranscriptLoader._parse_filename("transcript.md")

        assert speaker == "Unknown"
        assert title == "Transcript"

    def test_parse_filename_empty(self):
        """Test handling of empty filename."""
        speaker, title = TranscriptLoader._parse_filename(".md")

        assert speaker == "Unknown"
        assert title == "Untitled"


class TestTranscriptSearch:
    """Unit tests for transcript search functionality."""

    @pytest.fixture
    def loader(self) -> TranscriptLoader:
        """Create a TranscriptLoader instance."""
        return TranscriptLoader()

    def test_search_text_finds_matches(self, loader: TranscriptLoader):
        """Test that search finds matching lines."""
        results = loader.search_text("agents")

        assert len(results) > 0
        for transcript, matches in results:
            assert isinstance(transcript, Transcript)
            assert len(matches) > 0
            assert all("agent" in m.lower() for m in matches)

    def test_search_text_case_insensitive(self, loader: TranscriptLoader):
        """Test case-insensitive search."""
        results_lower = loader.search_text("agents", case_sensitive=False)
        results_upper = loader.search_text("AGENTS", case_sensitive=False)

        # Should find same transcripts
        transcripts_lower = {t.filename for t, _ in results_lower}
        transcripts_upper = {t.filename for t, _ in results_upper}
        assert transcripts_lower == transcripts_upper

    def test_search_text_case_sensitive(self, loader: TranscriptLoader):
        """Test case-sensitive search."""
        results_exact = loader.search_text("The", case_sensitive=True)
        results_wrong = loader.search_text("THE", case_sensitive=True)

        # Case sensitive search for "The" should find more than "THE"
        # (since transcripts use standard capitalization)
        matches_exact = sum(len(m) for _, m in results_exact)
        matches_wrong = sum(len(m) for _, m in results_wrong)
        assert matches_exact >= matches_wrong

    def test_search_text_no_matches(self, loader: TranscriptLoader):
        """Test search with no matches."""
        results = loader.search_text("xyznonexistentterm123")
        assert results == []


class TestEvalDatasetLoader:
    """Unit tests for eval dataset loading."""

    def test_load_eval_dataset_default(self):
        """Test loading default eval dataset."""
        dataset = load_eval_dataset()

        assert isinstance(dataset, EvalDataset)
        assert dataset.version == "1.0.0"
        assert dataset.count >= 10  # Should have 10-15 questions

    def test_load_eval_dataset_examples(self):
        """Test that loaded examples have required fields."""
        dataset = load_eval_dataset()

        for example in dataset.examples:
            assert example.id.startswith("eval_")
            assert len(example.question) >= 10
            assert len(example.reference_answer) >= 10
            assert len(example.expected_sections) >= 1
            assert example.difficulty_level is not None
            assert example.question_type is not None

    def test_load_eval_dataset_nonexistent(self, tmp_path: Path):
        """Test error handling for missing dataset file."""
        with pytest.raises(FileNotFoundError):
            load_eval_dataset(tmp_path / "nonexistent.json")

    def test_eval_dataset_by_difficulty(self):
        """Test filtering examples by difficulty."""
        from evals.schemas.task import DifficultyLevel

        dataset = load_eval_dataset()

        easy = dataset.by_difficulty(DifficultyLevel.EASY)
        medium = dataset.by_difficulty(DifficultyLevel.MEDIUM)
        hard = dataset.by_difficulty(DifficultyLevel.HARD)

        # Should have examples at each level
        assert len(easy) > 0
        assert len(medium) > 0
        assert len(hard) > 0

        # Total should match
        assert len(easy) + len(medium) + len(hard) == dataset.count

    def test_eval_dataset_by_type(self):
        """Test filtering examples by question type."""
        from evals.schemas.task import QuestionType

        dataset = load_eval_dataset()

        factual = dataset.by_type(QuestionType.FACTUAL)
        analytical = dataset.by_type(QuestionType.ANALYTICAL)
        opinion = dataset.by_type(QuestionType.OPINION)

        # Should have examples of each type
        assert len(factual) > 0
        assert len(analytical) > 0
        assert len(opinion) > 0

        # Total should match
        assert len(factual) + len(analytical) + len(opinion) == dataset.count


class TestTranscriptDataclass:
    """Unit tests for Transcript dataclass."""

    def test_transcript_creation(self):
        """Test creating a Transcript instance."""
        transcript = Transcript(
            filename="test.md",
            speaker="Test Speaker",
            title="Test Title",
            text="Test content",
            file_path="/path/to/test.md",
        )

        assert transcript.filename == "test.md"
        assert transcript.speaker == "Test Speaker"
        assert transcript.title == "Test Title"
        assert transcript.text == "Test content"
        assert transcript.file_path == "/path/to/test.md"

    def test_transcript_equality(self):
        """Test Transcript equality comparison."""
        t1 = Transcript("a.md", "Speaker", "Title", "Text", "/path")
        t2 = Transcript("a.md", "Speaker", "Title", "Text", "/path")
        t3 = Transcript("b.md", "Speaker", "Title", "Text", "/path")

        assert t1 == t2
        assert t1 != t3
