"""Render captured terminal output to SVG for README previews.

Why SVG:
- GitHub renders SVG inline from the repo (no camo, no CDN).
- Text stays selectable and legible at any zoom.
- A few KB per file; regenerable from a one-liner.

Usage:
    python scripts/make_preview_svg.py \\
        --input /tmp/agent_demo.txt \\
        --output docs/preview/agent-demo.svg \\
        --title "python scripts/agent_demo.py"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Layout constants tuned for 14px monospace (advance width ~0.6em).
CHAR_W = 8.4
LINE_H = 20.0
FONT_SIZE = 14
PAD = 24
TITLE_BLOCK_H = 36

FONT_STACK = (
    "ui-monospace, SFMono-Regular, Menlo, Consolas, "
    "'Liberation Mono', 'Courier New', monospace"
)
BG = "#0d1117"        # GitHub dark canvas
FG = "#e6edf3"        # GitHub dark default text
TITLE_FG = "#8b949e"  # muted
SEP = "#30363d"


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_svg(
    text: str,
    *,
    title: str | None = None,
    char_w: float = CHAR_W,
    line_h: float = LINE_H,
    font_size: int = FONT_SIZE,
    pad: int = PAD,
) -> str:
    """Return an SVG string. Deterministic for a given input."""
    lines = text.rstrip("\n").split("\n") if text.strip() else []

    max_len = max((len(line) for line in lines), default=0)
    width = int(pad * 2 + max_len * char_w)
    title_h = TITLE_BLOCK_H if title else 0
    height = int(pad * 2 + title_h + len(lines) * line_h)

    out: list[str] = []
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" '
        f'font-family="{FONT_STACK}">'
    )
    out.append(f'<rect width="{width}" height="{height}" rx="10" fill="{BG}"/>')

    if title:
        out.append(
            f'<text x="{pad}" y="{pad + 18}" fill="{TITLE_FG}" '
            f'font-size="13">{_escape(title)}</text>'
        )
        out.append(
            f'<line x1="0" y1="{pad + 28}" x2="{width}" y2="{pad + 28}" '
            f'stroke="{SEP}" stroke-width="1"/>'
        )

    y0 = pad + title_h
    for i, line in enumerate(lines):
        y = int(y0 + (i + 1) * line_h - 4)
        out.append(
            f'<text x="{pad}" y="{y}" fill="{FG}" font-size="{font_size}" '
            f'xml:space="preserve">{_escape(line)}</text>'
        )

    out.append("</svg>")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="Path to plain text (no ANSI). '-' reads stdin.")
    ap.add_argument("--output", required=True, help="Path to write SVG.")
    ap.add_argument("--title", default=None,
                    help="Optional title bar text (e.g. the command).")
    args = ap.parse_args(argv)

    text = sys.stdin.read() if args.input == "-" else Path(args.input).read_text()
    svg = render_svg(text, title=args.title)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(svg)
    print(f"wrote {args.output} ({len(svg)} bytes, {svg.count(chr(10))} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
