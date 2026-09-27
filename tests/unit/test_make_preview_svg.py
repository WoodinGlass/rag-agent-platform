"""Shape tests for the SVG preview renderer. Offline, stdlib only."""
from __future__ import annotations

import re

from scripts.make_preview_svg import render_svg


def _w(svg: str) -> int:
    return int(re.search(r'width="(\d+)"', svg).group(1))


def _h(svg: str) -> int:
    return int(re.search(r'height="(\d+)"', svg).group(1))


def test_render_minimal():
    svg = render_svg("hello world")
    assert svg.startswith("<svg ")
    assert svg.rstrip().endswith("</svg>")
    assert "hello world" in svg


def test_render_with_title():
    svg = render_svg("body", title="cmd --flag")
    assert "cmd --flag" in svg
    assert "body" in svg


def test_render_escapes_xml_specials():
    svg = render_svg('a & b < c > d " e')
    assert "&amp;" in svg
    assert "&lt;" in svg
    assert "&gt;" in svg
    assert "&quot;" in svg
    # raw angle brackets would break XML parsing
    assert "<c>" not in svg


def test_render_empty_is_still_valid_svg():
    svg = render_svg("")
    assert svg.startswith("<svg ")
    assert svg.rstrip().endswith("</svg>")


def test_render_preserves_leading_whitespace():
    svg = render_svg("    indented")
    assert 'xml:space="preserve"' in svg
    assert "    indented" in svg


def test_render_multiline_produces_one_text_per_line():
    svg = render_svg("a\nb\nc")
    assert svg.count("<text ") == 3


def test_render_title_adds_one_more_text():
    svg = render_svg("a\nb", title="cmd")
    # 1 title text + 2 body texts
    assert svg.count("<text ") == 3


def test_render_width_scales_with_longest_line():
    assert _w(render_svg("x" * 100)) > _w(render_svg("x"))


def test_render_height_scales_with_line_count():
    assert _h(render_svg("\n".join("x" for _ in range(50)))) > _h(render_svg("one"))


def test_render_deterministic():
    a = render_svg("same", title="x")
    b = render_svg("same", title="x")
    assert a == b
