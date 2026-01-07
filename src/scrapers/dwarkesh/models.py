"""Pydantic models for Dwarkesh Podcast transcript data."""

from datetime import datetime
from pydantic import BaseModel, Field


class ParsedTurn(BaseModel):
    """Single speaker turn from transcript."""

    speaker: str
    start_time_seconds: int | None = None
    text: str
    section_title: str | None = None
    ord: int = 0

    @property
    def timestamp_display(self) -> str:
        """Format timestamp as HH:MM:SS for display."""
        if self.start_time_seconds is None:
            return ""
        hours, remainder = divmod(self.start_time_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class ParsedSection(BaseModel):
    """Section header with timestamp."""

    title: str
    timestamp_seconds: int

    @property
    def timestamp_display(self) -> str:
        """Format timestamp as HH:MM:SS for display."""
        hours, remainder = divmod(self.timestamp_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class Episode(BaseModel):
    """Complete parsed episode."""

    url: str
    slug: str
    title: str
    guest: str | None = None
    published_at: datetime | None = None
    turns: list[ParsedTurn] = Field(default_factory=list)
    sections: list[ParsedSection] = Field(default_factory=list)
    raw_transcript: str | None = None  # Full transcript text for debugging

    @property
    def total_turns(self) -> int:
        """Total number of speaker turns."""
        return len(self.turns)

    @property
    def speakers(self) -> list[str]:
        """Unique speakers in the episode."""
        return list(dict.fromkeys(turn.speaker for turn in self.turns))

    def to_json_dict(self) -> dict:
        """Convert to JSON-serializable dict for Modal volume storage."""
        return {
            "url": self.url,
            "slug": self.slug,
            "title": self.title,
            "guest": self.guest,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "turns": [turn.model_dump() for turn in self.turns],
            "sections": [section.model_dump() for section in self.sections],
            "raw_transcript": self.raw_transcript,
        }

    @classmethod
    def from_json_dict(cls, data: dict) -> "Episode":
        """Create Episode from JSON dict loaded from Modal volume."""
        return cls(
            url=data["url"],
            slug=data["slug"],
            title=data["title"],
            guest=data.get("guest"),
            published_at=datetime.fromisoformat(data["published_at"]) if data.get("published_at") else None,
            turns=[ParsedTurn(**t) for t in data.get("turns", [])],
            sections=[ParsedSection(**s) for s in data.get("sections", [])],
            raw_transcript=data.get("raw_transcript"),
        )


class EpisodeMetadata(BaseModel):
    """Lightweight episode metadata for discovery phase."""

    url: str
    slug: str
    title: str
    published_at: datetime | None = None
