"""Scrapers for various podcast and transcript sources."""

# Lazy imports to avoid circular dependencies
__all__ = ["DwarkeshScraper", "DwarkeshParser"]


def __getattr__(name: str):
    if name == "DwarkeshScraper":
        from src.scrapers.dwarkesh.scraper import DwarkeshScraper
        return DwarkeshScraper
    elif name == "DwarkeshParser":
        from src.scrapers.dwarkesh.parser import DwarkeshParser
        return DwarkeshParser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
