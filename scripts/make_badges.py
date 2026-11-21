"""Write small flat SVG badges to docs/images/badges/ (no external service).

    python scripts/make_badges.py
"""
import os
from xml.sax.saxutils import escape

from asp_benchmark.paths import DOCS_DIR

OUT = str(DOCS_DIR / "images" / "badges")
LABEL = "#5a6b8c"  # label half: slate blue (not near black)
BADGES = {  # name: (label, value, value-half colour); one distinct hue per badge, no red or black
    "python": ("Python", "3.12", "#3776ab"), "pandas": ("pandas", "data handling", "#7a5bd0"),
    "numpy": ("NumPy", "arrays, RNG", "#4d9de0"), "scipy": ("SciPy", "statistics", "#2a9d8f"),
    "statsmodels": ("statsmodels", "OLS, MixedLM", "#8e6bd8"), "lifelines": ("lifelines", "KM, Cox PH", "#e88c2a"),
    "linearmodels": ("linearmodels", "PanelOLS", "#3aa675"), "scikit-learn": ("scikit-learn", "GBM, learners", "#f2a33a"),
    "matplotlib": ("matplotlib", "figures", "#2f8f9d"), "gemini": ("Gemini API", "google-genai", "#5b8def"),
    "jupyter": ("Jupyter", "notebooks", "#f37726"), "tests": ("tests", "14 behavioural", "#1b9e8a"),
    "synthetic": ("data", "synthetic, known truth", "#c98512"), "license": ("license", "MIT", "#6b7280"),
    "models": ("LLMs compared", "6 Gemini models", "#7b4fd6"),
    "github-actions": ("GitHub Actions", "pytest on every push", "#2c7be5"), "pytest": ("pytest", "12 tests", "#0a9bb5"),
}


def badge(label, value, color):
    lw, vw = 7 * len(label) + 14, 7 * len(value) + 14
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{lw + vw}" height="22" role="img" aria-label="{escape(label)}: {escape(value)}">'
            f'<rect width="{lw}" height="22" rx="3" fill="{LABEL}"/><rect x="{lw}" width="{vw}" height="22" rx="3" fill="{color}"/>'
            f'<rect x="{lw - 3}" width="6" height="22" fill="{color}"/>'
            f'<g fill="#fff" font-family="Verdana,DejaVu Sans,sans-serif" font-size="11" text-anchor="middle">'
            f'<text x="{lw / 2}" y="15">{escape(label)}</text><text x="{lw + vw / 2}" y="15">{escape(value)}</text></g></svg>')


os.makedirs(OUT, exist_ok=True)
for name, (l, v, c) in BADGES.items():
    open(os.path.join(OUT, f"{name}.svg"), "w").write(badge(l, v, c))
print("badges:", sorted(os.listdir(OUT)))
