#!/usr/bin/env python3
"""
generate_pdf.py
Utility to sanitize a LaTeX file produced by nbconvert by removing fancyhdr/header lines
and other header/footer commands that insert the notebook filename.

Usage: python3 tools/generate_pdf.py notebook.tex
"""
import sys, re

def sanitize(path):
    s = open(path, 'r', encoding='utf-8').read()
    s_new = s
    # Remove any fancyhdr blocks, commands and header/footer definitions
    # Remove \usepackage{fancyhdr} ... \makeatother blocks
    s_new = re.sub(r"\\usepackage\{fancyhdr\}.*?\\makeatother\n", "", s_new, flags=re.S)
    # Remove any \fancypagestyle{...}{...} blocks
    s_new = re.sub(r"\\fancypagestyle\{.*?\}\{.*?\}\n", "", s_new, flags=re.S)
    # Remove single-line header/footer commands
    patterns = [
        r"\\pagestyle\{[^}]*\}",
        r"\\fancyhf\{[^}]*\}",
        r"\\lhead\{[^}]*\}",
        r"\\chead\{[^}]*\}",
        r"\\rhead\{[^}]*\}",
        r"\\lfoot\{[^}]*\}",
        r"\\cfoot\{[^}]*\}",
        r"\\rfoot\{[^}]*\}",
        r"\\fancyhead\[[^\]]*\]\{[^}]*\}",
        r"\\fancyfoot\[[^\]]*\]\{[^}]*\}",
        r"\\markboth\{[^}]*\}\{[^}]*\}",
        r"\\markright\{[^}]*\}"
    ]
    for p in patterns:
        s_new = re.sub(p + r"\n", "\n", s_new)
        s_new = re.sub(p, "", s_new)
    # Ensure we remove any \\thispagestyle{...} on title page
    s_new = re.sub(r"\\thispagestyle\{[^}]*\}", "", s_new)
    # Remove title/author/date and disable \maketitle so the notebook name
    # doesn't appear in the document header or title area.
    s_new = re.sub(r"\\title\{.*?\}\s*", "", s_new, flags=re.S)
    s_new = re.sub(r"\\author\{.*?\}\s*", "", s_new, flags=re.S)
    s_new = re.sub(r"\\date\{.*?\}\s*", "", s_new, flags=re.S)
    s_new = re.sub(r"\\maketitle\b", "", s_new)
    # Insert robust override before \begin{document} to force no header/footer
    # This redefines the plain page style to empty and sets pagestyle empty.
    if re.search(r"\\begin\{document\}", s_new):
        # Use real newlines ("\n") instead of literal backslash-n so LaTeX is valid.
        # Inject a counter to disable automatic section numbering (removes 1.1, 1.2 ...)
        replacement = (
            '\\makeatletter\\def\\ps@plain{\\ps@empty}\\makeatother\n'
            '\\pagestyle{empty}\n'
            '\\setcounter{secnumdepth}{0}\n'
            '% --- injected by sanitizer: clear running headers/footers and disable section numbers\n'
            '\\begin{document}'
        )
    # Use a direct string replace to avoid regex backreference handling and satisfy linters.
    s_new = s_new.replace("\\begin{document}", replacement, 1)
    if s_new == s:
        print("No changes made to", path)
    else:
        open(path, 'w', encoding='utf-8').write(s_new)
        print("Sanitized", path)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 tools/generate_pdf.py notebook.tex")
        sys.exit(1)
    sanitize(sys.argv[1])
