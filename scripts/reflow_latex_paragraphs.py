#!/usr/bin/env python3
"""Reflow LaTeX prose so each source paragraph occupies one physical line.

The formatter deliberately leaves the preamble, displayed mathematics, floats,
TikZ, tables, listings and other code-like environments unchanged.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


PROTECTED_ENVIRONMENTS = {
    "figure", "figure*", "table", "table*", "tabular", "tabularx",
    "longtable", "tikzpicture", "equation", "equation*", "align",
    "align*", "aligned", "gather", "gather*", "multline", "multline*",
    "split", "cases", "matrix", "pmatrix", "bmatrix", "vmatrix",
    "verbatim", "Verbatim", "lstlisting", "minted", "algorithm",
    "algorithmic", "thebibliography",
}

STANDALONE_COMMAND = re.compile(
    r"^\\(?:chapter|chapter\*|section|section\*|subsection|subsection\*|"
    r"subsubsection|subsubsection\*|label|begin|end|clearpage|newpage|"
    r"pagebreak|nopagebreak|vspace|vspace\*|hspace|hspace\*|medskip|"
    r"smallskip|bigskip|printbibliography|addcontentsline|setquotestyle|"
    r"pagenumbering|tableofcontents|listoffigures|listoftables|"
    r"printglossary|glsaddall|maketitle|input|include|appendix|"
    r"bibliography|bibliographystyle|nocite|centering|raggedright|"
    r"raggedleft|thispagestyle|pagestyle|setcounter|addtocounter|"
    r"renewcommand|newcommand|providecommand|DeclareRobustCommand)\b"
)


def protected_begin(line: str) -> str | None:
    match = re.match(r"^\s*\\begin\{([^}]+)\}", line)
    return match.group(1) if match and match.group(1) in PROTECTED_ENVIRONMENTS else None


def protected_end(line: str) -> str | None:
    match = re.match(r"^\s*\\end\{([^}]+)\}", line)
    return match.group(1) if match and match.group(1) in PROTECTED_ENVIRONMENTS else None


def reflow(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    paragraph: list[str] = []
    protected_stack: list[str] = []
    in_document = False
    display_math = False

    def flush() -> None:
        if paragraph:
            output.append(" ".join(part.strip() for part in paragraph if part.strip()))
            paragraph.clear()

    for raw in lines:
        stripped = raw.strip()

        if not in_document:
            output.append(raw)
            if stripped == r"\begin{document}":
                in_document = True
            continue

        if protected_stack:
            output.append(raw)
            begin = protected_begin(raw)
            if begin:
                protected_stack.append(begin)
            end = protected_end(raw)
            if end and protected_stack and protected_stack[-1] == end:
                protected_stack.pop()
            continue

        begin = protected_begin(raw)
        if begin:
            flush()
            output.append(raw)
            protected_stack.append(begin)
            continue

        if stripped in {r"\[", r"$$"}:
            flush()
            output.append(raw)
            display_math = True
            continue
        if display_math:
            output.append(raw)
            if stripped in {r"\]", r"$$"}:
                display_math = False
            continue

        if not stripped:
            flush()
            if output and output[-1] != "":
                output.append("")
            continue

        if stripped.startswith("%"):
            flush()
            output.append(raw)
            continue

        if stripped == r"\end{document}":
            flush()
            output.append(raw)
            continue

        if STANDALONE_COMMAND.match(stripped):
            flush()
            output.append(raw)
            continue

        if stripped.startswith(r"\item"):
            flush()
            paragraph.append(stripped)
            continue

        # Prose, including paragraphs beginning with \noindent, \textbf,
        # \gls or \cite, is accumulated until a real paragraph boundary.
        paragraph.append(stripped)

    flush()
    while len(output) > 1 and output[-1] == "" and output[-2] == "":
        output.pop()
    return "\n".join(output) + "\n"


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: reflow_latex_paragraphs.py INPUT OUTPUT")
    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    destination.write_text(reflow(source.read_text(encoding="utf-8")), encoding="utf-8")


if __name__ == "__main__":
    main()
