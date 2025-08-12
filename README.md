 # (N)ormalizing flow (A)lgorithms beyond (Z)ero-variance training

Dicrete and continuous normalizing flows for conditional and unconditional density estimation and simulation based inference. In addition to standard training, fully Bayesian uncertainty quantification and model comprison tools are provided. The main goal is to marginalize over the epistemic and aleatoric variance in flow predictions which is particularly relevant for sparse and noisy training datasets while also performing model selection among arcitectures to avoid over-fitting. Exact (direct posterior sampling) and approximate (stochastic variational inference wuth importance sampling) Bayesian flows and Monte Carlo dropout scemes are implemented.

## Instalation:
```
git clone https://github.com/AnaryaRay1/naz.git
cd naz
conda env create --name yourcoolname --file environment.yml
pip install --upgrade -r pip_requirements.txt
pip install .
```

## Citation

```
@article{Ray:2025xdo,
    author = "Ray, Anarya",
    title = "{Emulating compact binary population synthesis simulations with robust uncertainty quantification and model comparison: Bayesian normalizing flows}",
    eprint = "2506.05657",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    month = "6",
    year = "2025"
}
```

## Example usage
See ```examples/papers/2506.05657```.
