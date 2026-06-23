# TokenTriage Manuscript

arXiv-style manuscript source for *TokenTriage: Harm-Aware Token Routing for Clinical
Language Model Inference*.

- `main.tex` — paper draft (abstract, contributions, method, results with embedded figures,
  limitations, reproducibility, release macros).
- `references.bib` — BibTeX entries for the early-exit related work.
- `figures/` — vector PDF figures, generated from the project-page SVGs.
- `main.pdf` — the built paper.

## Build

The PDF builds with [Tectonic](https://tectonic-typesetting.github.io/) (self-contained, no
TeX install required):

```bash
tectonic main.tex
```

Or with a traditional TeX distribution:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Figures

The figures embedded in the paper are rendered from the project-page SVG assets and converted
to vector PDF with `rsvg-convert`:

```bash
for f in gap pareto architecture validation critical_mix examples; do
  rsvg-convert -f pdf -o figures/$f.pdf ../project-page/assets/$f.svg
done
```

Regenerate the underlying SVGs from the experiment artifacts first if the data changes:

```bash
python ../scripts/create_project_figures.py
```

## Before public submission

Update the release macros at the top of `main.tex` to the real URLs:

```tex
\newcommand{\repoURL}{\url{https://github.com/DIGlabUAB/token-triage}}
\newcommand{\projectURL}{\url{https://diglabuab.github.io/token-triage}}
```
