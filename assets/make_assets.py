"""Generate the README's SVGs, each in a light and a dark variant.

Usage: python3 assets/make_assets.py

Backgrounds are transparent; the README picks the variant matching GitHub's theme with
<picture><source media="(prefers-color-scheme: dark)">.
"""
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).parent
SANS = 'font-family="-apple-system,BlinkMacSystemFont,\'Segoe UI\',Helvetica,Arial,sans-serif"'
THEMES = {  # text, muted, track, accent — tuned against GitHub's #ffffff / #0d1117
    "light": ("#1f2328", "#656d76", "#e7e9ec", "#4f46e5"),
    "dark": ("#e6edf3", "#8d96a0", "#262c36", "#8b85ff"),
}


def write(name, w, h, body, label):
    for theme, colours in THEMES.items():
        (OUT / f"{name}-{theme}.svg").write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{escape(label)}">\n{body(*colours)}\n</svg>\n')


def header(text, muted, track, accent):
    # A request forks into three tiers; the default (middle) path is the lit one.
    fork = "".join(
        f'<path d="M58 120 C 100 120, 100 {y}, 142 {y}" fill="none" stroke="{c}" stroke-width="{sw}" '
        f'stroke-linecap="round"/><circle cx="150" cy="{y}" r="{r}" fill="{c}"/>'
        for y, c, sw, r in ((72, track, 3, 6), (168, track, 3, 9), (120, accent, 4, 7.5)))
    return (f'<line x1="14" y1="120" x2="58" y2="120" stroke="{accent}" stroke-width="4" stroke-linecap="round"/>'
            f'<circle cx="14" cy="120" r="7" fill="{text}"/>{fork}'
            f'<text x="210" y="138" {SANS} font-size="76" letter-spacing="-2" fill="{text}">'
            f'<tspan font-weight="300">Smart</tspan><tspan font-weight="700">Route</tspan></text>'
            f'<text x="214" y="184" {SANS} font-size="22" fill="{muted}">'
            'Cheapest model that can answer · checked · escalated only when wrong</text>')


BARS = [  # share of always-GPT-4's cost to reach 95% of its accuracy (RouterBench, ladder B)
    ("Always GPT-4", 100),
    ("SmartRoute router, no checker", 69),
    ("Best fixed model (Zero router)", 58),
    ("Router + learned checker + peers", 55),
    ("SmartRoute + Jev answer check", 22.7),
]
X0, X1, TOP, GAP = 400, 1090, 150, 52


def bench(text, muted, track, accent):
    out = [f'<text x="0" y="40" {SANS} font-size="26" font-weight="600" fill="{text}">'
           'Cost to reach 95% of GPT-4 accuracy</text>',
           f'<text x="0" y="76" {SANS} font-size="17" fill="{muted}">RouterBench · 10,956 held-out prompts · '
           'Mixtral-8x7B → GPT-3.5 → GPT-4 · lower is better</text>']
    for i, (label, pct) in enumerate(BARS):
        y, ours = TOP + GAP * i, i == len(BARS) - 1
        fill, weight = (accent, "600") if ours else (muted, "400")
        out += [f'<text x="0" y="{y + 6}" {SANS} font-size="18" font-weight="{weight}" '
                f'fill="{text if ours else muted}">{escape(label)}</text>',
                f'<rect x="{X0}" y="{y - 5}" width="{X1 - X0}" height="10" rx="5" fill="{track}"/>',
                f'<rect x="{X0}" y="{y - 5}" width="{(X1 - X0) * pct / 100:.1f}" height="10" rx="5" fill="{fill}"/>',
                f'<text x="1200" y="{y + 6}" text-anchor="end" {SANS} font-size="18" font-weight="600" '
                f'fill="{accent if ours else text}">{pct:g}%</text>']
    return "\n".join(out)


write("header", 1200, 240, header, "SmartRoute")
write("benchmark", 1200, TOP + GAP * len(BARS) - 20, bench,
      "Cost to reach 95% of GPT-4 accuracy: " + ", ".join(f"{l} {p:g}%" for l, p in BARS))
print("wrote", *sorted(p.name for p in OUT.glob("*.svg")))
