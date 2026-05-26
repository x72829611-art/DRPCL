# DRPCL

Deep Representation Learning for Proximal Causal Learning (DRPCL).

## Environment

- Python 3.12
- `torch==2.5.1`
- `numpy==1.26.4`
- `scipy==1.16.3`
- `pandas==2.3.2`
- `scikit-learn==1.4.2`
- `matplotlib==3.10.8`

## Data

The code expects dataset files under:
- `data/ihdp/`
- `data/jobs/`
- `data/twins/`

## Reproduction

Run from the repository root:

```bash
bash run_ihdp_exp.sh
```

```bash
bash run_jobs_exp.sh
```

```bash
bash run_twins_exp.sh
```

Results are written to `experiment/results/<knob>/` by default.
