[![Paper](https://img.shields.io/badge/Paper-LaTeX-red?style=flat-square&logo=latex&logoColor=white)](paper/main.tex)
[![Python Versions](https://img.shields.io/badge/Python-3.13-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)

Semantic Structure Defines Relation Geometry in Transformers
------

> **Abstract:** Prior work shows that relations expressed more often in pretraining data are more likely to admit linear decoders in language-model representations. We find that corpus frequency by itself does not determine this geometry, but we identify another factor, *semantic structure* (the attributes that subjects and objects share), that plays an important role in shaping geometry. We construct two controlled synthetic settings, A and B. The Transformer in A sees each fact of attribute-defined relations with low frequency, yet a probe diagnoses the relations as highly linear and the model predicts correct objects for held-out subjects with 0.635 recall, demonstrating good generalization. In contrast, a frequency-matched random relation (with no attribute structure) remains at chance under the same probe. The Transformer in B memorizes a random relation sampled ten times as often as each of ten controls with rich semantic structure. The random relation's accuracy is 0.9998, yet an affine fit explains only 0.375 of target variance, compared to 0.995 for the structured relations. These results give controlled counterexamples to a frequency-only account and point to the importance of semantic structure for relation geometry. Finally, in our analysis of relations in the LMEnt-1B model, we similarly find only a weak association between corpus frequency and affine-decoder faithfulness, suggesting that frequency is only one of many factors that shape relation geometry in both synthetic and natural data.

This repository contains the code, experimental pipelines, and saved measurements for the paper “Semantic Structure Defines Relation Geometry in Transformers” by Jonas Rohweder and Hinrich Schütze.

[CIS, LMU Munich](https://www.cis.lmu.de/) | [UKP Lab, TU Darmstadt](https://www.ukp.tu-darmstadt.de/) | [MCML](https://mcml.ai/)

## Prerequisites

This project uses [uv](https://github.com/astral-sh/uv) as the Python package manager. For the CPU workflow, use Linux and Python 3.13.11. Install dependencies and set up the virtual environment:

```bash
uv sync --locked
```

Building the paper and generating figures require Cairo, TeX Live, and `latexmk`. Exact reproduction of the paper's synthetic measurements uses the separate pinned Docker environment on a Ryzen 9 7950X3D with one RTX 4090 and NVIDIA driver 610.57.04; see the [reproduction documentation](docs/reproduction.md).

## Build the paper from saved artifacts

Compile the manuscript using the committed numerical results and figures:

```bash
latexmk -cd -pdf paper/main.tex
```

This writes `paper/main.pdf` without training models or generating a `results/` directory. To also package the arXiv source archive:

```bash
python3 paper/repro/build_arxiv.py
```

## Run full reproduction

On the certified RTX setup, build the pinned environments and run both experiments, analyses, figures, paper compilation, and the numerical audit:

```bash
docker build --network host -t factual-recall:reproducible .
docker build --network host -f docker/rtx/Dockerfile -t factual-recall:rtx-runtime docker/rtx
docker build --network host -f Dockerfile.rtx -t factual-recall:rtx .
python scripts/run_rtx.py --output /app/results/full
```

Outputs are saved under `results/full/`. LMEnt analysis uses saved observations; original pretrained-model inference and corpus extraction are external inputs.

## Run checks

Run the CPU test suite:

```bash
uv run --locked pytest -q
```

Exercise the complete pipeline with a shortened CPU run:

```bash
uv run --locked reproduce --smoke
```

Smoke runs check the pipeline and do not reproduce paper values. See the [reproduction documentation](docs/reproduction.md) for RTX checks, resume commands, outputs, baseline evidence, and verification scope. List individual pipeline stages with `uv run --locked reproduce --help`.

## Cite

If you found our data or code helpful, please cite our paper:

```bibtex
@misc{rohweder2026semantic,
  title={Semantic Structure Defines Relation Geometry in Transformers},
  author={Rohweder, Jonas and Schütze, Hinrich},
  year={2026},
  howpublished={Preprint},
  url={https://github.com/jonasrohw/affine-structure-defines-linear-geometry}
}
```

## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.
