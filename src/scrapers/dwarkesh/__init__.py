"""Dwarkesh Podcast scraper and parser."""

__all__ = ["DwarkeshParser", "DwarkeshScraper"]


def __getattr__(name: str):
    if name == "DwarkeshParser":
        from src.scrapers.dwarkesh.parser import DwarkeshParser
        return DwarkeshParser
    elif name == "DwarkeshScraper":
        from src.scrapers.dwarkesh.scraper import DwarkeshScraper
        return DwarkeshScraper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
