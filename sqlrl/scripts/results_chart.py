# Usage: python scripts/results_chart.py
# Draws docs/results-light.svg and docs/results-dark.svg from the final reports and the Spider dev
# report in engineering/:
# one row per split, one track per model, released (ring) to fine-tuned (dot).
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REPORTS = {"1B": HERE / "engineering/release-1b-pipeline-report.md", "2B": HERE / "engineering/release-2b-report.md"}
ROWS = [
    ("Never-trained templates (test set)", "Test set", "9 unseen types, scored once"),
    ("Never-trained templates (check set)", "Check set", "5 unseen types"),
    ("Never-trained templates (selection set)", "Selection set", "5 unseen types"),
    ("Trained templates, unseen parameters and data", "Trained types", "new values and data"),
    ("Spider dev", "Spider dev", "20 real schemas, not trained"),
]
SPIDER = HERE / "engineering/spider-dev-report.md"
THEMES = {
    "light": dict(bg="#ffffff", ink="#25272c", ink2="#6b6f78", ink3="#8f939b", track="#eeeff1", grid="#e2e3e6",
                  base="#9aa1ab", **{"1B": "#2a78d6", "2B": "#eb6834"}),
    "dark": dict(bg="#16171a", ink="#ececee", ink2="#a6a8ae", ink3="#7c7f86", track="#232429", grid="#2c2d33",
                 base="#5d6169", **{"1B": "#3987e5", "2B": "#d95926"}),
}


def scores(path):
    out = {}
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 3 and cells[0] in {r[0] for r in ROWS}:
            out[cells[0]] = tuple(float(re.match(r"([\d.]+)%", c).group(1)) for c in cells[1:])
    return out


def spider(path):
    # Test-suite pass@1 from the first table: {model: (released, fine-tuned)}
    rates = {}
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 4 and re.match(r"(Released|Fine-tuned) [12]B$", cells[0]):
            rates[cells[0]] = float(re.match(r"([\d.]+)%", cells[1]).group(1))
    return {m: (rates[f"Released {m}"], rates[f"Fine-tuned {m}"]) for m in ("1B", "2B")}


def draw(t, data):
    W, L, R, top, row_h, track_h = 760, 230, 130, 84, 78, 26
    H = top + row_h * len(ROWS) + 12
    x = lambda v: L + v / 100 * (W - L - R)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif">',
         f'<rect width="{W}" height="{H}" rx="10" fill="{t["bg"]}"/>',
         f'<text x="20" y="30" font-size="15" font-weight="600" fill="{t["ink"]}">Released and fine-tuned MiniCPM5, pass rate on databases they never trained on</text>']
    lx = 20
    for label, kind in [("Released", "ring"), ("Fine-tuned 1B", "1B"), ("Fine-tuned 2B", "2B")]:
        if kind == "ring":
            s.append(f'<circle cx="{lx + 5}" cy="47" r="4.5" fill="{t["bg"]}" stroke="{t["base"]}" stroke-width="2"/>')
        else:
            s.append(f'<circle cx="{lx + 5}" cy="47" r="5" fill="{t[kind]}"/>')
        s.append(f'<text x="{lx + 15}" y="51" font-size="12" fill="{t["ink2"]}">{label}</text>')
        lx += 30 + len(label) * 7
    for tick in (0, 25, 50, 75, 100):
        anchor = "start" if tick == 0 else "end" if tick == 100 else "middle"
        s.append(f'<text x="{x(tick):.1f}" y="{top - 2}" font-size="11" text-anchor="{anchor}" fill="{t["ink3"]}">{tick}%</text>')
    for i, (key, name, note) in enumerate(ROWS):
        y0 = top + 8 + i * row_h
        s.append(f'<text x="20" y="{y0 + 18}" font-size="13.5" font-weight="600" fill="{t["ink"]}">{name}</text>')
        s.append(f'<text x="20" y="{y0 + 36}" font-size="11.5" fill="{t["ink3"]}">{note}</text>')
        for j, model in enumerate(("1B", "2B")):
            before, after = data[model][key]
            cy = y0 + 13 + j * track_h
            s.append(f'<text x="{L - 12}" y="{cy + 4}" font-size="12" font-weight="600" text-anchor="end" fill="{t["ink2"]}">{model}</text>')
            s.append(f'<rect x="{L}" y="{cy - 1}" width="{W - L - R}" height="2" rx="1" fill="{t["track"]}"/>')
            for tick in (25, 50, 75):
                s.append(f'<rect x="{x(tick):.1f}" y="{cy - 5}" width="1" height="10" fill="{t["grid"]}"/>')
            s.append(f'<rect x="{x(before):.1f}" y="{cy - 1.5}" width="{x(after) - x(before):.1f}" height="3" rx="1.5" fill="{t[model]}" opacity="0.45"/>')
            s.append(f'<circle cx="{x(before):.1f}" cy="{cy}" r="5" fill="{t["bg"]}" stroke="{t["base"]}" stroke-width="2"/>')
            s.append(f'<circle cx="{x(after):.1f}" cy="{cy}" r="6" fill="{t[model]}" stroke="{t["bg"]}" stroke-width="2"/>')
            s.append(f'<text x="{W - 20}" y="{cy + 4}" font-size="12.5" text-anchor="end" fill="{t["ink2"]}">'
                     f'{before:.1f}% → <tspan font-weight="600" fill="{t["ink"]}">{after:.1f}%</tspan></text>')
    s.append("</svg>")
    return "\n".join(s) + "\n"


data = {m: scores(p) for m, p in REPORTS.items()}
for m, pair in spider(SPIDER).items():
    data[m]["Spider dev"] = pair
for name, theme in THEMES.items():
    (HERE.parent / f"docs/results-{name}.svg").write_text(draw(theme, data))
print({m: d["Never-trained templates (test set)"] for m, d in data.items()})
