"""Write small flat SVG badges to docs/images/badges/ (no external service)."""
import os
from xml.sax.saxutils import escape

OUT = "docs/images/badges"
TEAL, BLUE, PURPLE, AMBER, GREY, SLATE = "#1b9e8a", "#2a6fdb", "#7b4fd6", "#c98512", "#6b7280", "#3b4a6b"
BADGES = {
    "python": ("Python", "3.12", BLUE), "pandas": ("pandas", "data", SLATE), "numpy": ("NumPy", "arrays", SLATE),
    "scipy": ("SciPy", "stats", SLATE), "statsmodels": ("statsmodels", "OLS, MixedLM", PURPLE),
    "lifelines": ("lifelines", "KM, Cox PH", PURPLE), "linearmodels": ("linearmodels", "PanelOLS", PURPLE),
    "scikit-learn": ("scikit-learn", "GBM, learners", PURPLE), "matplotlib": ("matplotlib", "figures", SLATE),
    "gemini": ("Gemini API", "google-genai", BLUE), "jupyter": ("Jupyter", "notebooks", SLATE),
    "tests": ("tests", "14 behavioural", TEAL), "synthetic": ("data", "synthetic, known truth", AMBER),
    "license": ("license", "MIT", GREY), "models": ("LLMs compared", "6 Gemini models", BLUE),
}


def badge(label, value, color):
    lw, vw = 7 * len(label) + 14, 7 * len(value) + 14
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{lw + vw}" height="22" role="img" aria-label="{escape(label)}: {escape(value)}">'
            f'<rect width="{lw}" height="22" rx="3" fill="#3b4a6b"/><rect x="{lw}" width="{vw}" height="22" rx="3" fill="{color}"/>'
            f'<rect x="{lw - 3}" width="6" height="22" fill="{color}"/>'
            f'<g fill="#fff" font-family="Verdana,DejaVu Sans,sans-serif" font-size="11" text-anchor="middle">'
            f'<text x="{lw / 2}" y="15">{escape(label)}</text><text x="{lw + vw / 2}" y="15">{escape(value)}</text></g></svg>')


os.makedirs(OUT, exist_ok=True)
for name, (l, v, c) in BADGES.items():
    open(os.path.join(OUT, f"{name}.svg"), "w").write(badge(l, v, c))
print("badges:", sorted(os.listdir(OUT)))
