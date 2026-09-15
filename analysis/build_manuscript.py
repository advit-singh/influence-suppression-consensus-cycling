"""Convert the cleaned manuscript text into LaTeX section files."""

from __future__ import annotations

import re
from pathlib import Path

from analysis.fix_latex_math import fix_math

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "_audit" / "new_paper.txt"
OUT_DIR = ROOT / "paper" / "sections"

CITE_MAP = {
    "2": "degroot1974reaching",
    "3": "friedkin1990social",
    "4": "acemoglu2013opinion",
    "5": "hajnal1958weak",
    "6": "leblanc2013resilient",
}

SECTION_MARKERS = [
    ("introduction.tex", r"^1\. Introduction"),
    ("related_work.tex", r"^2\. Related Work"),
    ("preliminaries.tex", r"^3\. Preliminaries"),
    ("model.tex", r"^4\. Model Formulation"),
    ("theory.tex", r"^5\. Theoretical Analysis"),
    ("methods.tex", r"^6\. Experimental Design"),
    ("results.tex", r"^7\. Results and Discussion"),
    ("conclusion.tex", r"^8\. Conclusion"),
]


def load_paragraphs() -> list[str]:
    text = SOURCE.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    cleaned = []
    skip = False
    for paragraph in paragraphs:
        if paragraph.startswith("=" * 20):
            skip = not skip
            continue
        if skip:
            continue
        if paragraph.startswith("[INSTRUCTIONS FOR LATEX"):
            continue
        if re.match(r"^(Tab \d+|introduction|related work|results and discussion)$", paragraph, re.I):
            continue
        cleaned.append(paragraph)
    return cleaned


def convert_citations(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        keys = []
        for token in match.group(1).split(","):
            token = token.strip()
            if token in CITE_MAP:
                keys.append(CITE_MAP[token])
        if not keys:
            return ""
        return "\\cite{" + ",".join(keys) + "}"

    text = re.sub(r"\[cite:\s*([0-9,\s]+)\]", repl, text)
    text = text.replace("{0, 1, 2, \\dots}", r"\{0,1,2,\ldots\}")
    text = text.replace("{20, 50, 100, 200}", r"\{20,50,100,200\}")
    text = text.replace("{0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0}", r"\{0.0,0.1,0.2,0.3,0.5,0.7,0.9,1.0\}")
    text = text.replace("{0.0, 0.1, 0.2, 0.5}", r"\{0.0,0.1,0.2,0.5\}")
    text = text.replace("{N, N/2, N/5, N/10}", r"\{N,N/2,N/5,N/10\}")
    text = text.replace("368,993", "372,960")
    text = text.replace("368,991", "372,960")
    return text


def is_heading(paragraph: str) -> bool:
    return bool(re.match(r"^\d+(\.\d+)*[\.\)]?\s+", paragraph))


def paragraph_to_latex(paragraph: str) -> str:
    paragraph = convert_citations(paragraph)
    if paragraph.startswith("Comparison of Consensus"):
        return "\\input{sections/comparison_table}\n"
    if paragraph.startswith("Summary of Experimental Parameter Grid"):
        return "\\input{sections/parameter_table}\n"
    if paragraph.startswith("Cross-Model Performance Summary"):
        return "\\input{sections/cross_model_table}\n"
    if paragraph.startswith("Model / Baseline"):
        return ""
    if paragraph in {
        "Validated Runs",
        "Regime 1: Biased Convergence (%)",
        "Regime 1B: Fixed Dissensus (%)",
        "Regime 2: Slow Mixing (%)",
        "Regime 3: Cycling (%)",
        "Mean Consensus Error ($E_{cons}$)",
        "Mean Temporal Variance ($\\text{Var}_{temp}$)",
        "Mean Product Hajnal ($\\delta(M)$)",
    }:
        return ""
    if re.match(r"^(Proposed|Symmetric|Fixed|Anti-Expert|Inverse|Stubborn|\$|[0-9])", paragraph):
        # Drop the markdown-style table body; LaTeX table is injected separately.
        if "%" in paragraph or "Control" in paragraph or "Ablation" in paragraph:
            return ""
    if is_heading(paragraph):
        title = re.sub(r"^\d+(\.\d+)*\s+", "", paragraph)
        title = title.replace("$", "")
        if re.match(r"^\d+\.\d+", paragraph):
            return f"\\subsection{{{title}}}\n"
        return f"\\section{{{title}}}\n"
    if paragraph.startswith("- "):
        items = [line[2:].strip() for line in paragraph.split("\n") if line.startswith("- ")]
        body = "\n".join(f"  \\item {item}" for item in items)
        return f"\\begin{{itemize}}\n{body}\n\\end{{itemize}}\n"
    return paragraph + "\n\n"


def split_sections(paragraphs: list[str]) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {name: [] for name, _ in SECTION_MARKERS}
    current = None
    for paragraph in paragraphs:
        for name, pattern in SECTION_MARKERS:
            if re.match(pattern, paragraph):
                current = name
                break
        if current:
            sections[current].append(paragraph)
    return sections


def write_sections() -> None:
    paragraphs = load_paragraphs()
    sections = split_sections(paragraphs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, content in sections.items():
        latex = fix_math("".join(paragraph_to_latex(p) for p in content))
        (OUT_DIR / name).write_text(latex, encoding="utf-8")


if __name__ == "__main__":
    write_sections()
    print(f"Wrote LaTeX sections to {OUT_DIR}")
