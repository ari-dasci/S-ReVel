# ReVel

[![PyPI](https://img.shields.io/pypi/v/revel-xai.svg)](https://pypi.org/project/revel-xai/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> A framework to measure local linear explanations of black-box models.

ReVel takes an explanation method that fits a local linear surrogate — LIME, SHAP, or a
random baseline — and scores it with five metrics, so explanation methods can be compared
with numbers instead of eyeballing heatmaps. It was introduced in
*REVEL Framework to Measure Local Linear Explanations for Black-Box Models: Deep Learning
Image Classification Case Study*.

## What it measures

Given a black-box model `B`, an instance `x` and a local linear explanation `L` — a linear
regressor over a segmentation of `x` — ReVel computes:

| Metric | Question it answers |
| --- | --- |
| `local_concordance` | How far is the surrogate's prediction from the black-box prediction **on the instance itself**? |
| `local_fidelity` | Same distance, but **over the neighbourhood** of the instance. |
| `prescriptivity` | Same distance **on the neighbour `X_h` of the instance**, i.e. away from `x`. |
| `conciseness` | How concentrated the explanation is: built from the importance matrix `A` (`F x C`) using the norm of each feature's importance vector. |
| `robustness` | How stable the importance matrices are across repeated fits of the same explanation method. |

## Installation

The package is published on PyPI as `revel-xai` (Python 3.8+):

```bash
pip install revel-xai
```

From source:

```bash
git clone https://github.com/ari-dasci/S-ReVel.git
cd S-ReVel
pip install .
```

## Quick start

```python
from ReVel import ReVel
from ReVel.perturbations import get_perturbation
from ReVel.LLEs import get_xai_model

# 1. How the image is segmented and perturbed to build the neighbourhood.
perturbation = get_perturbation(
    name="square",          # or "quickshift"
    dim=14,                 # 14x14 grid of segments
    num_classes=10,
    final_size=(224, 224),
)

# 2. The explanation method to evaluate: "LIME", "SHAP" or "RANDOM".
explainer = get_xai_model(
    name="LIME",
    perturbation=perturbation,
    max_examples=1000,
    sigma=3,
)

# 3. Segment the instance. `image` is an HxWxC numpy array.
segments = explainer.perturbation.segmentation_fn(image)

# 4. Evaluate the explanation of that instance.
revel = ReVel(
    model_f=classify_input_space,    # callable: image -> class probabilities
    model_g=classify_feature_space,  # callable: segmented/perturbed image -> class probabilities
    instance=image,
    lle=explainer,
    n_classes=10,
    segments=segments,
)

scores = revel.evaluate(times=5)
print(scores.mean())
```

`evaluate(times)` refits the explanation `times` times and returns a `pandas.DataFrame`
with one row per run and five columns: `conciseness`, `local_fidelity`,
`local_concordance`, `prescriptivity` and `robustness`.

A complete end-to-end run — EfficientNet classifier, CIFAR-10 and ImageNet-200 loaders,
segmentation, perturbation and plots — is in
[`docs/source/notebooks/HowToMeasureAnExplanation.ipynb`](docs/source/notebooks/HowToMeasureAnExplanation.ipynb).

For a script you can run in one command, with no dataset and no download, see
[`examples/quickstart.py`](examples/quickstart.py):

```bash
python examples/quickstart.py
```

It builds a small CNN with random weights, explains a random image with LIME and prints
the five metrics, so you can see the input/output contract of every piece before wiring in
your own model.

## Documentation

Full API documentation, including the tutorial notebooks, is at
<https://revel-xai.readthedocs.io/en/latest/>.

## Paper and citation

If you use ReVel in your research, please cite:

> Sevillano-García, I., Luengo, J., Herrera, F. (2023).
> *REVEL Framework to Measure Local Linear Explanations for Black-Box Models: Deep Learning
> Image Classification Case Study.* International Journal of Intelligent Systems, 2023.
> <https://doi.org/10.1155/2023/8068569>

```bibtex
@article{sevillano2023revel,
  title   = {REVEL Framework to Measure Local Linear Explanations for Black-Box Models:
             Deep Learning Image Classification Case Study},
  author  = {Sevillano-Garc{\'i}a, Iv{\'a}n and Luengo, Juli{\'a}n and Herrera, Francisco},
  journal = {International Journal of Intelligent Systems},
  volume  = {2023},
  year    = {2023},
  doi     = {10.1155/2023/8068569}
}
```

## License

GNU Affero General Public License v3.0 — see the [LICENSE](LICENSE) file.

## Authors

- **Iván Sevillano-García** — [isevillano@ugr.es](mailto:isevillano@ugr.es)
- Julián Luengo
- Francisco Herrera

Research Group: [Andalusian Research Institute in Data Science and Computational
Intelligence (DaSCI)](https://dasci.es), University of Granada, Spain.
