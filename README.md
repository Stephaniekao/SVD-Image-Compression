# Image Compression with the SVD

**DS5020 Final Project, Iteration 3**

**Team Members:** Chengyuan Liu, Wan Chi Kao
**Project Leader (Iteration 3):** Wan Chi Kao

## Project Overview

In this project, we study how the Singular Value Decomposition (SVD) can be used for image compression. The main idea is to keep only the largest singular values, so we can reduce the storage cost without losing too much information.

In Iteration 2, we derived the mathematics of the SVD and checked it with a simulation. In Iteration 3, we implement the SVD **from scratch with NumPy only** (power iteration with deflation on $A^\top A$) and verify it in three ways:

1. A small worked example ($2 \times 2$) computed step by step with exact fractions.
2. A simulation where the true answer is known, compared with the truth and with `numpy.linalg.svd`.
3. The real image, compared with `numpy.linalg.svd`.

`numpy.linalg` is used only to check our answers, never inside `src/methods.py`.

The Iteration 2 version of this repository is saved as the release **`iteration-2`**.

## Project Structure

```text
.
├── data/
│   └── photo.jpeg
├── figures/
│   ├── original_image.png
│   ├── rgb_histogram.png
│   ├── grayscale_image.png
│   └── sim_error_vs_size.png
├── src/
│   ├── methods.py            # from-scratch SVD, one function per algorithm step
│   ├── simulate.py           # simulate_low_rank(): the Iteration 2 simulation as a function
│   ├── small_example.py      # the 2 x 2 example, step by step
│   ├── check_simulation.py   # simulation test
│   ├── run_real.py           # real image and library comparison
│   ├── image_analysis.py     # data exploration and load_image()
│   ├── test_methods.py       # quick tests of each function in methods.py
│   └── simulation.py         # Iteration 2 simulation check (kept for reference)
├── docs/
│   ├── iteration2.pdf
│   ├── iteration3.pdf
│   └── project_plan.xlsx
└── README.md
```

## Requirements

We use Python 3 and the following libraries:

- NumPy
- Matplotlib
- Pillow (only to read the image file)

```bash
python -m pip install numpy matplotlib pillow
```

## How to Run

Run every command from the main repository folder, so that figures are saved in `figures/`.

### 1. Small worked example

```bash
python src/small_example.py
```

TODO (Chengyuan): what the script prints.

### 2. Simulation test

```bash
python src/check_simulation.py
```

Runs the from-scratch SVD on the Iteration 2 simulation (120 × 80, rank 5 plus noise 0.01, seed 0) in three parts:

- **Part 1:** compares $\sigma_i$, $U$ and $V$ (signs aligned) with `numpy.linalg.svd`, and checks $\|A - A_5\|_F^2 = \sum_{i>5} \sigma_i^2$.
- **Part 2:** compares the rank-5 approximation with the noise-free truth $L$ for four sizes and saves `figures/sim_error_vs_size.png`.
- **Part 3:** stress test with $k = 8$, showing slow convergence when $\sigma_{i+1}/\sigma_i$ is close to 1.

### 3. Real image

```bash
python src/run_real.py
```

TODO (Chengyuan): arguments, what the script prints, and which figures it saves.

### Other scripts

```bash
python src/test_methods.py                   # test each function in methods.py on the 2 x 2 example
python src/simulate.py                       # check that the simulation matches Iteration 2 exactly
python src/image_analysis.py data/photo.jpeg # image statistics and figures
```

## Results

### Simulation test

Comparison with `numpy.linalg.svd` (120 × 80):

| Quantity | Max difference |
|---|---:|
| $\sigma_1, \dots, \sigma_5$ | 7.11e-14 |
| $U$ (signs aligned) | 2.61e-12 |
| $V$ (signs aligned) | 3.03e-12 |
| Tail sum $\|A - A_5\|_F^2$ vs $\sum_{i>5}\sigma_i^2$ | 1.78e-15 |

Comparison with the truth $L$:

| Size | $\|A_5 - L\|_F / \|L\|_F$ | $\|A - L\|_F / \|L\|_F$ |
|---|---:|---:|
| 60 × 40 | 2.01e-03 | 4.28e-03 |
| 120 × 80 | 1.48e-03 | 4.78e-03 |
| 480 × 320 | 7.33e-04 | 4.50e-03 |
| 1200 × 800 | 4.56e-04 | 4.49e-03 |

The log–log slope is −0.50, so the error shrinks like $1/\sqrt{m}$.

![Simulation error against size](figures/sim_error_vs_size.png)

Stress test ($k = 8$): the noise singular values $\sigma_6, \sigma_7, \sigma_8$ are almost equal ($\sigma_8/\sigma_7 \approx 0.99$), so $v_8$ needs about 1,900 iterations, while $v_1, \dots, v_5$ need at most about 250.

### Small worked example

TODO (Chengyuan)

### Real image

The image is 724 × 1086. In Iteration 2 we analyzed the RGB channels; in Iteration 3 we convert it to grayscale ($0.299R + 0.587G + 0.114B$) and scale it to $[0, 1]$, as the Iteration 3 handout asks.

TODO (Chengyuan): library comparison, compressed images for k = 5, 20, 50 and the 95%-energy k, storage cost.

## Reproducibility

We use `np.random.default_rng(0)` to fix the random seed. `simulate_low_rank()` draws the random numbers in the same order as the Iteration 2 script, so it produces exactly the same matrix.

## Project Report

Our report is written in LaTeX on Overleaf.

- [Iteration 2 Report (PDF)](docs/iteration2.pdf)
- [Iteration 3 Report (PDF)](docs/iteration3.pdf)

The updated Excel project plan is in the `docs` folder.


## Chengyuan Liu Iteration 3 additions

### Small worked example (Q1, Q2)

```bash
python src/small_example.py
```

Computes the SVD of A = [[3, 0], [4, 5]] step by step with exact fractions, checks it by substitution, then runs `my_svd_power` on the same matrix and compares with the exact values and `numpy.linalg.svd`.

### Real data (Q5)

```bash
python src/run_real.py
```

Loads `data/photo.jpeg` as a grayscale matrix in [0, 1], runs `my_svd_power` (k = 50), compares with `numpy.linalg.svd`, prints the energy, error and storage cost for k = 2 (95% energy), 5, 16 (99% energy), 20 and 50, and saves `figures/real_energy.png` and `figures/real_reconstructions.png`.

### Assumption checks

```bash
python src/check_assumptions.py
```

Checks the four Iteration 2 assumptions on the simulation first, then on the real image, and saves `figures/assumption_checks.png`.

`src/verify.py` holds the shared `report(name, mine, ref)` helper from the handout, so every comparison prints in the same format.
