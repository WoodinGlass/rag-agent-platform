# Preview assets

SVG renderings of real terminal output, embedded in the top-level
README. SVGs are used because:

- GitHub renders them inline from the repo (no CDN, no camo).
- Text stays selectable and legible at any zoom.
- Files are a few KB; the repo stays light.

## Naming

| File | Source command |
|---|---|
| `agent-demo.svg` | `python scripts/agent_demo.py` |
| `stream-demo.svg` | `python scripts/stream_demo.py --n 5 --duplicates 2` |
| `benchmark-1k.svg` | `cat benchmarks/results/scale_1000.md` (mirrors the CLI title) |
| `provider-smoke.svg` | `pytest tests/integration/test_provider_smoke.py -m provider -v` |

## Regenerate

```bash
# 1. capture plain text (strip color; no ANSI)
python scripts/agent_demo.py > /tmp/agent-demo.txt

# 2. render
python scripts/make_preview_svg.py \
  --input /tmp/agent-demo.txt \
  --output docs/preview/agent-demo.svg \
  --title "python scripts/agent_demo.py"
Or run the batch cell that ships in the milestone notes. Nothing here
depends on a display; it works in CI, Colab, or a headless box.
