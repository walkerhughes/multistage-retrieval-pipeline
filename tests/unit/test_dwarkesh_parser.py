"""Tests for Dwarkesh Podcast transcript parser."""

import pytest

from src.scrapers.dwarkesh.parser import (
    DwarkeshParser,
    parse_timestamp_to_seconds,
    SPEAKER_PATTERN,
    SECTION_PATTERN,
)


class TestTimestampParsing:
    """Tests for timestamp parsing."""

    def test_parse_timestamp_basic(self):
        """Test basic timestamp parsing."""
        assert parse_timestamp_to_seconds("0", "0", "0") == 0
        assert parse_timestamp_to_seconds("0", "0", "30") == 30
        assert parse_timestamp_to_seconds("0", "1", "0") == 60
        assert parse_timestamp_to_seconds("1", "0", "0") == 3600

    def test_parse_timestamp_complex(self):
        """Test complex timestamp values."""
        # 1:30:45 = 3600 + 1800 + 45 = 5445
        assert parse_timestamp_to_seconds("1", "30", "45") == 5445
        # 2:15:30 = 7200 + 900 + 30 = 8130
        assert parse_timestamp_to_seconds("2", "15", "30") == 8130


class TestSpeakerPattern:
    """Tests for speaker pattern matching."""

    def test_matches_standard_format(self):
        """Test matching standard speaker format."""
        text = "**Dwarkesh Patel** _00:00:00_"
        match = SPEAKER_PATTERN.search(text)
        assert match is not None
        assert match.group("speaker") == "Dwarkesh Patel"
        assert match.group("hours") == "00"
        assert match.group("minutes") == "00"
        assert match.group("seconds") == "00"

    def test_matches_single_digit_hours(self):
        """Test matching single-digit hours."""
        text = "**Guest Name** _0:15:30_"
        match = SPEAKER_PATTERN.search(text)
        assert match is not None
        assert match.group("speaker") == "Guest Name"
        assert match.group("hours") == "0"
        assert match.group("minutes") == "15"
        assert match.group("seconds") == "30"

    def test_matches_double_digit_hours(self):
        """Test matching double-digit hours."""
        text = "**Speaker** _02:30:45_"
        match = SPEAKER_PATTERN.search(text)
        assert match is not None
        assert match.group("hours") == "02"

    def test_no_match_without_timestamp(self):
        """Test no match when timestamp is missing."""
        text = "**Speaker Name**"
        match = SPEAKER_PATTERN.search(text)
        assert match is None


class TestSectionPattern:
    """Tests for section header pattern matching."""

    def test_matches_with_en_dash(self):
        """Test matching section with en-dash."""
        text = "[(00:15:30) – AGI timelines]"
        match = SECTION_PATTERN.search(text)
        assert match is not None
        assert match.group("title") == "AGI timelines"
        assert match.group("hours") == "00"
        assert match.group("minutes") == "15"
        assert match.group("seconds") == "30"

    def test_matches_with_hyphen(self):
        """Test matching section with regular hyphen."""
        text = "[(01:00:00) - Future of AI]"
        match = SECTION_PATTERN.search(text)
        assert match is not None
        assert match.group("title") == "Future of AI"

    def test_matches_single_digit_hour(self):
        """Test matching single-digit hour in section."""
        text = "[(1:30:00) – Topic]"
        match = SECTION_PATTERN.search(text)
        assert match is not None
        assert match.group("hours") == "1"


class TestDwarkeshParser:
    """Tests for DwarkeshParser class."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return DwarkeshParser()

    def test_parse_simple_transcript(self, parser):
        """Test parsing a simple two-turn transcript."""
        content = """
**Dwarkesh Patel** _00:00:00_
Welcome to the show! Today we have a special guest.

**Guest Name** _00:00:15_
Thank you for having me. It's great to be here.
"""
        turns, sections = parser.parse_transcript(content)

        assert len(turns) == 2
        assert turns[0].speaker == "Dwarkesh Patel"
        assert turns[0].start_time_seconds == 0
        assert "Welcome to the show" in turns[0].text
        assert turns[0].ord == 0

        assert turns[1].speaker == "Guest Name"
        assert turns[1].start_time_seconds == 15
        assert "Thank you for having me" in turns[1].text
        assert turns[1].ord == 1

    def test_parse_transcript_with_sections(self, parser):
        """Test parsing transcript with section headers."""
        content = """
[(00:00:00) – Introduction]

**Dwarkesh Patel** _00:00:00_
Let's start with introductions.

[(00:05:00) – Main Topic]

**Dwarkesh Patel** _00:05:00_
Now let's get into the main topic.

**Guest** _00:05:30_
Sure, I'd love to discuss this.
"""
        turns, sections = parser.parse_transcript(content)

        assert len(sections) == 2
        assert sections[0].title == "Introduction"
        assert sections[0].timestamp_seconds == 0
        assert sections[1].title == "Main Topic"
        assert sections[1].timestamp_seconds == 300

        assert len(turns) == 3
        assert turns[0].section_title == "Introduction"
        assert turns[1].section_title == "Main Topic"
        assert turns[2].section_title == "Main Topic"

    def test_parse_empty_transcript(self, parser):
        """Test parsing empty content."""
        turns, sections = parser.parse_transcript("")
        assert len(turns) == 0
        assert len(sections) == 0

    def test_parse_no_speakers(self, parser):
        """Test parsing content without speaker markers."""
        content = "Just some regular text without any speaker markers."
        turns, sections = parser.parse_transcript(content)
        assert len(turns) == 0

    def test_clean_turn_text_removes_sections(self, parser):
        """Test that section headers are removed from turn text."""
        content = """
**Speaker** _00:00:00_
Some text before. [(00:01:00) – New Section] Some text after.
"""
        turns, _ = parser.parse_transcript(content)

        assert len(turns) == 1
        # Section header should be removed
        assert "[(00:01:00)" not in turns[0].text
        assert "New Section]" not in turns[0].text
        assert "Some text before" in turns[0].text
        assert "Some text after" in turns[0].text

    def test_timestamp_display(self, parser):
        """Test timestamp display formatting."""
        content = "**Speaker** _01:30:45_ Hello"
        turns, _ = parser.parse_transcript(content)

        assert len(turns) == 1
        assert turns[0].timestamp_display == "01:30:45"

    def test_handles_long_transcript(self, parser):
        """Test parsing a longer transcript with multiple turns."""
        turns_data = []
        for i in range(10):
            speaker = "Dwarkesh Patel" if i % 2 == 0 else "Guest"
            minutes = i * 5
            turns_data.append(f"**{speaker}** _00:{minutes:02d}:00_\nTurn {i} content here.")

        content = "\n\n".join(turns_data)
        turns, sections = parser.parse_transcript(content)

        assert len(turns) == 10
        for i, turn in enumerate(turns):
            assert turn.ord == i
            expected_seconds = i * 5 * 60
            assert turn.start_time_seconds == expected_seconds


class TestParsedTurn:
    """Tests for ParsedTurn model."""

    def test_timestamp_display_none(self):
        """Test timestamp display with None timestamp."""
        from src.scrapers.dwarkesh.models import ParsedTurn

        turn = ParsedTurn(
            speaker="Test",
            start_time_seconds=None,
            text="Hello",
        )
        assert turn.timestamp_display == ""

    def test_timestamp_display_formatted(self):
        """Test timestamp display formatting."""
        from src.scrapers.dwarkesh.models import ParsedTurn

        turn = ParsedTurn(
            speaker="Test",
            start_time_seconds=3661,  # 1:01:01
            text="Hello",
        )
        assert turn.timestamp_display == "01:01:01"


class TestEpisode:
    """Tests for Episode model."""

    def test_speakers_list(self):
        """Test getting unique speakers from episode."""
        from src.scrapers.dwarkesh.models import Episode, ParsedTurn

        episode = Episode(
            url="https://example.com/p/test",
            slug="test",
            title="Test Episode",
            turns=[
                ParsedTurn(speaker="Dwarkesh Patel", text="Q1"),
                ParsedTurn(speaker="Guest", text="A1"),
                ParsedTurn(speaker="Dwarkesh Patel", text="Q2"),
                ParsedTurn(speaker="Guest", text="A2"),
            ],
        )

        speakers = episode.speakers
        assert len(speakers) == 2
        assert "Dwarkesh Patel" in speakers
        assert "Guest" in speakers
        # Order should be preserved (first occurrence)
        assert speakers[0] == "Dwarkesh Patel"

    def test_json_round_trip(self):
        """Test JSON serialization/deserialization."""
        from datetime import datetime
        from src.scrapers.dwarkesh.models import Episode, ParsedTurn, ParsedSection

        original = Episode(
            url="https://example.com/p/test",
            slug="test",
            title="Test Episode",
            guest="Guest Name",
            published_at=datetime(2024, 1, 15, 12, 0, 0),
            turns=[
                ParsedTurn(
                    speaker="Dwarkesh Patel",
                    start_time_seconds=0,
                    text="Hello",
                    section_title="Intro",
                    ord=0,
                ),
            ],
            sections=[
                ParsedSection(title="Intro", timestamp_seconds=0),
            ],
        )

        # Round-trip through JSON
        json_dict = original.to_json_dict()
        restored = Episode.from_json_dict(json_dict)

        assert restored.url == original.url
        assert restored.slug == original.slug
        assert restored.title == original.title
        assert restored.guest == original.guest
        assert restored.published_at == original.published_at
        assert len(restored.turns) == len(original.turns)
        assert restored.turns[0].speaker == original.turns[0].speaker
        assert len(restored.sections) == len(original.sections)
