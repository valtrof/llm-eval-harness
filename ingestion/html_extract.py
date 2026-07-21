"""Strips real documentation pages down to their main content, discarding nav bars, sidebars,
cookie banners, and other boilerplate that a naive "grab all the text" extraction would keep."""

import trafilatura


def extract_html(raw_html: bytes) -> str:
    """Return the page's main content as markdown, or an empty string if extraction fails.
    Markdown (not plain text) is used specifically so headings keep their '#' markers — that's
    what lets the ingestion step detect section boundaries later instead of guessing at them."""
    text = trafilatura.extract(
        raw_html,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
        favor_recall=True,
    )
    return text or ""
