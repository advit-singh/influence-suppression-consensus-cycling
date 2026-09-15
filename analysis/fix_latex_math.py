"""Repair common LaTeX math artifacts from the Word export."""

from __future__ import annotations

import re
from pathlib import Path

SECTION_DIR = Path(__file__).resolve().parents[1] / "paper" / "sections"

AUGMENTED_MATRIX = (
    r"\begin{bmatrix} (1 - \gamma_{\text{stub}}) \tilde{W}(t) & \gamma_{\text{stub}} x(0) "
    r"\\ \mathbf{0}^T & 1 \end{bmatrix}"
)


def fix_cases(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        body = match.group(1)
        body = re.sub(
            r"(?<![\\])\\(?=\s+(?:w'|w_|\\sum|\(1|1\.0|1 -|\(1 -))",
            r"\\\\",
            body,
        )
        return f"\\begin{{cases}}{body}\\end{{cases}}"

    return re.sub(r"\\begin\{cases\}(.*?)\\end\{cases\}", repl, text, flags=re.S)


def fix_subsection_titles(text: str) -> str:
    return re.sub(
        r"\\subsection\{([^}]*)\(\\gamma_\{stub\}",
        r"\\subsection{\1($\\gamma_{\\text{stub}}",
        text,
    )


def fix_missing_stars(text: str) -> str:
    text = re.sub(r"(\w)\^(\$|\)|,|\||\\cite|=|\s+\\)", r"\1^*\2", text)
    text = re.sub(r"\\mu\^\(", r"\\mu*(", text)
    text = re.sub(r"\\sigma\^\(", r"\\sigma*(", text)
    text = re.sub(r"x_([123AB])_\^(\()", r"x_\1^*\2", text)
    text = re.sub(r"x_([123AB])\^\(", r"x_\1^*(", text)
    text = re.sub(r"x_A\^\(", r"x_A^*(", text)
    text = re.sub(r"\\gamma_\{stub\}\^\$", r"\\gamma_{\\text{stub}}^*", text)
    return text


def fix_norms_and_seminorms(text: str) -> str:
    text = re.sub(r"\|x\|\{osc\}", r"|x|_{\\text{osc}}", text)
    text = re.sub(r"\|W x\|\{osc\}", r"|Wx|_{\\text{osc}}", text)
    text = re.sub(r"\|x\|\{osc\}", r"|x|_{\\text{osc}}", text)
    text = re.sub(r"\|x\(0\)\|\{osc\}", r"|x(0)|_{\\text{osc}}", text)
    text = re.sub(r"\|x\([^)]+\)\|\{osc\}", lambda m: m.group(0).replace("{osc}", "_{\\text{osc}}"), text)
    text = re.sub(r"\\max\{i\}", r"\\max_i", text)
    text = re.sub(r"\\min_\{i\}", r"\\min_i", text)
    text = re.sub(
        r"\|\(1 - \\gamma_\{\\text\{stub\}\}\) \\tilde\{W\}_A\|\\infty",
        r"\\|(1 - \\gamma_{\\text{stub}}) \\tilde{W}_A\\|_\\infty",
        text,
    )
    return text


def fix_broken_bmatrices(text: str) -> str:
    broken = (
        r"\begin{bmatrix} (1 - \\gamma_{stub}) \\tilde{W}(t) & \\gamma_{stub} x(0) "
        r"\\ \\mathbf\{0\}^T & 1 \end{bmatrix}"
    )
    return text.replace(broken, AUGMENTED_MATRIX)


def fix_math(text: str) -> str:
    replacements = [
        (r"\\tilde\{w\}\{([0-9][a-z])\}", r"\\tilde{w}_{\1}"),
        (r"\\tilde\{w\}\{([a-z]{2})\}", r"\\tilde{w}_{\1}"),
        (r"\\tilde\{w\}\{([0-9]+\\cdot)\}", r"(\\tilde{w}_A)_{\1}"),
        (r"w'\{([0-9][a-z]|[a-z]{2})\}", r"w'_{\1}"),
        (r"w\{([0-9][a-z]|[a-z]{2})\}", r"w_{\1}"),
        (r"\\gamma\{stub\}", r"\\gamma_{\\text{stub}}"),
        (r"\\text\{Var\}j", r"\\text{Var}_j"),
        (r"\\text\{MAD\}j", r"\\text{MAD}_j"),
        (r"\\bar\{x\}\{j,\s*t\}", r"\\bar{x}_{j,t}"),
        (r"\\sum\{\\tau", r"\\sum_{\\tau"),
        (r"\\sum\{i=1\}", r"\\sum_{i=1}"),
        (r"\\sum\{j=1\}", r"\\sum_{j=1}"),
        (r"\\sum\{k=1\}", r"\\sum_{k=1}"),
        (r"\\prod\{\\tau", r"\\prod_{\\tau"),
        (r"\\mathcal\{D\}([AB])\b", r"\\mathcal{D}_\1"),
        (r"\\tilde\{W\}\{([AB])\}", r"\\tilde{W}_\1"),
        (r"\\tilde\{W\}([AB])\b", r"\\tilde{W}_\1"),
        (r"\\mathbb\{R\}\{\\ge\s*0\}", r"\\mathbb{R}_{\\ge 0}"),
        (r"\\mu\^\$", r"\\mu^*"),
        (r"\\mu\^\|", r"\\mu^*|"),
        (r"\\sigma\^\$", r"\\sigma^*"),
        (r"\\sigma\^\|", r"\\sigma^*|"),
        (r"\$x_([AB])\^\$", r"$x_\1^*$"),
        (r"\\max\\big\(0,\s*;\s*", r"\\max\\big(0, "),
        (r"\\max\(0,\s*;\s*", r"\\max(0, "),
        (r"\\left\{", r"\\left\\{"),
        (r"\\right\}", r"\\right\\}"),
        (r"\\max\{i \\in V\}", r"\\max_{i \\in V}"),
        (r"\\text\{Var\}\{temp\}", r"\\text{Var}_{\\text{temp}}"),
        (r"\\sigma_\{\\infty\}\^2", r"\\sigma_{\\infty}^2"),
        (r"\|A\(t\)\|\\infty", r"\\|A(t)\\|_\\infty"),
        (r"for \$M \\in \{A, B\}\$", r"for $M \\in \\{A, B\\}$"),
        (r"\\mathcal\{D\}_A = \{x", r"\\mathcal{D}_A = \\{x"),
        (r"\\mathcal\{S\} = \{x", r"\\mathcal{S} = \\{x"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)

    text = fix_subsection_titles(text)
    text = fix_missing_stars(text)
    text = fix_norms_and_seminorms(text)
    text = fix_broken_bmatrices(text)
    text = fix_cases(text)

    text = text.replace("V = \\{1, 2, \\dots, N\\}", r"V = \{1, 2, \ldots, N\}")
    text = text.replace("K \\in \\left\\{ N, ; \\frac{N}{2}", r"K \in \left\{ N, \frac{N}{2}")
    text = text.replace(", ; ", ", ")
    text = text.replace("\\\\gamma_{stub}", r"\\gamma_{\text{stub}}")
    text = text.replace("\\\\tilde{W}", r"\\tilde{W}")
    text = text.replace("\\\\mathbf\\{0\\}", r"\\mathbf{0}")
    return text


def main() -> None:
    for path in SECTION_DIR.glob("*.tex"):
        original = path.read_text(encoding="utf-8")
        fixed = fix_math(original)
        if fixed != original:
            path.write_text(fixed, encoding="utf-8")
            print(f"fixed {path.name}")


if __name__ == "__main__":
    main()
