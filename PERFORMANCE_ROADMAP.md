# symWannier Wannierization Performance Roadmap

This document summarizes a practical roadmap for making `symWannier`
wannierization faster on multicore machines. It is intended for future
developers, users, or AI agents who need to continue the work without relying on
conversation context.

## Background

The target workflow is the Python wannierization path, typically invoked as:

```bash
python $SymWann/wannierize.py -S wannier > stdout
```

A representative previous calculation exists at:

```text
/home/user2/minkyu/computations/Co/pbesol-spn-kjpaw/wannier.sym
```

The user observed that WannierBerri benefits strongly from multicore execution,
while `symWannier` does not appear to scale similarly. The current decision is
to postpone Ray-style distributed/task parallelism and first improve the
single-node implementation.

## High-Level Assessment

Efficient multicore execution is possible, but the expected strategy differs
from WannierBerri's common interpolation/integration workload.

Wannierization has an outer iterative dependency:

1. compute current spread/gradient-like quantities,
2. update `Umat` and `Mmn`,
3. use the updated state in the next iteration.

This makes the outer iteration serial. The useful parallelism is mostly inside
each iteration:

- k-point independent diagonalizations and SVDs,
- blockwise matrix products,
- large BLAS-backed tensor contractions,
- selected file parsing and output operations.

The best first improvements are therefore local algorithmic and shared-memory
optimizations, not Ray.

## Main Code Locations

The most relevant files and routines are:

- `src/symwannier/wannierize.py`
  - `Wannierize.wannierization`
  - `Wannierize.calc_dw`
  - `Wannierize.calc_expdw`
  - `Wannierize.update_mmn`
  - `Wannierize.update_Mmn_by_Umat`
  - `Wannierize.dis_project`
  - `Wannierize.dis_extract`
  - `Wannierize.write_hr`
  - `Wannierize.write_tb`
- `src/symwannier/amn.py`
  - `Amn.Umat`
- `src/symwannier/mmn.py`
  - `Mmn._read_mmn`
  - `Mmn._mmn_full_klist`

## Roadmap

### 1. Establish a Profiling Baseline

Before changing algorithms, measure where time is actually spent on a real
calculation. Use the representative `wannier.sym` calculation above if possible.

Add or improve timers around:

- `calc_dw`
- `calc_expdw`
- `update_Mmn_by_Umat`
- `calc_omega`
- `calc_omega_detail`
- `dis_project`
- `dis_extract`
- `write_hr`
- `write_tb`
- `mmn`, `amn`, and `eig` reading

The goal is to separate:

- Python loop overhead,
- LAPACK/BLAS compute time,
- memory bandwidth pressure,
- text file I/O,
- repeated diagnostic output.

Also record problem size:

- `nk`
- `nb`
- `num_bands`
- `num_wann`
- whether disentanglement is enabled,
- `num_iter`,
- `dis_num_iter`,
- symmetry mode: `-s` or `-S`.

### 2. Remove Duplicate Diagonalization in `calc_expdw`

This should be the first real optimization.

Current `update_mmn` effectively computes:

```text
exp(dw)
exp(dw / 2)
exp(alpha * dw)
```

through separate calls to `calc_expdw`. Each call diagonalizes the same Hermitian
matrix up to a scalar factor. The eigenvectors are identical, and only the
eigenvalue scaling changes.

Refactor so that each k-point diagonalization is done once per iteration:

1. diagonalize `1j * dw[k]`,
2. reuse the eigenvectors,
3. construct exponentials for scale factors `1`, `0.5`, and later `alpha`.

Expected benefit:

- large if `calc_expdw` is a significant cost,
- reduces work before adding any parallel framework,
- lowers the amount of work that later parallel code must distribute.

Important detail:

- `alpha` is known only after evaluating the trial `omega1` and `omega2`, so
  store eigenvalues/eigenvectors or a compact object that can build `exp(c*dw)`
  after `alpha` is chosen.

### 3. Define BLAS/MKL Threading Policy

The environment may already use threaded BLAS/LAPACK. For example, NumPy linked
against MKL can use several threads inside `einsum`, `matmul`, `svd`, and
`eigh`.

Before adding Python-level parallelism, define a thread policy:

- For large dense contractions, allow BLAS/MKL threads.
- For many independent small k-point `eigh`/`svd` tasks, use multiple Python
  workers but limit each worker to one BLAS thread.
- Avoid oversubscription, where `N` Python workers each spawn `M` MKL threads.

Likely tools:

- environment variables:
  - `MKL_NUM_THREADS`
  - `OMP_NUM_THREADS`
  - `OPENBLAS_NUM_THREADS`
- `threadpoolctl` for local control inside Python sections.

Expected benefit:

- prevents multicore slowdown from nested threading,
- makes profiling reproducible,
- provides a stable foundation for later k-point parallelism.

### 4. Parallelize k-Point `eigh` and `svd` Work

After duplicate diagonalization is removed and threading policy is clear,
parallelize the independent k-point linear algebra.

Primary targets:

- `Wannierize.calc_expdw`
  - k-point diagonalization of `1j * dw[k]`.
- `Wannierize.dis_project`
  - k-point frozen/window correction diagonalization.
- `Wannierize.dis_extract`
  - k-point diagonalization of `zmat_nfroz`.
- `Amn.Umat`
  - k-point SVDs used to initialize `Umat`.

Implementation guidance:

- Do not dispatch one tiny task per k-point if matrices are small.
- Use k-blocks to reduce scheduling overhead.
- Keep arrays contiguous where possible.
- Use process or thread workers only after measuring which is better with the
  installed BLAS.
- If using processes, avoid copying large arrays repeatedly.

Candidate backends:

- `concurrent.futures`
- `joblib`
- a small internal block scheduler

Ray should remain a later option, not the first implementation.

### 5. Optimize `update_Mmn_by_Umat`

`update_Mmn_by_Umat` currently uses a large `einsum`:

```python
np.einsum("klm, kblp, kbpn->kbmn", ...)
```

This is mathematically compact, but performance depends heavily on the chosen
contraction path, memory layout, temporary arrays, and BLAS behavior.

Benchmark alternatives:

- current single `einsum`,
- two-stage contraction,
- batched `np.matmul`,
- k-blocked contractions to reduce peak memory,
- explicit control of temporary allocation.

Be careful with process parallelism here. `mmn` and `Umat` are large arrays, and
copying them into workers can erase any compute gain.

Expected benefit:

- potentially large for big `nk`, `nb`, or `num_wann`,
- likely memory-bandwidth sensitive,
- should be optimized with profiler data rather than guessed.

### 6. Reduce Unnecessary Per-Iteration Diagnostics

The current wannierization loop can print spreads and detailed omega components
every iteration. For large runs this may add nontrivial overhead and produce
large logs.

Improve this with runtime options:

- print detailed spreads every `N` iterations,
- compute `calc_omega_detail` only when requested or every `N` iterations,
- keep minimal `omega_tot` output for convergence tracking,
- add a verbose/debug mode for the current behavior.

Important:

- convergence logic must still use the required `omega` value,
- diagnostic suppression must not change numerical results.

### 7. Improve File I/O After Compute Bottlenecks

Large `.mmn`, `.amn`, `.eig`, `.hr`, and `.tb` text files can be expensive.
However, I/O should be optimized after confirming it is a bottleneck.

Possible improvements:

- faster parsing for `.mmn` and `.amn`,
- chunked parsing,
- cached binary intermediate format,
- buffered construction of output strings for `write_hr` and `write_tb`,
- avoid repeated `calc_irvec` if both `write_hr` and `write_tb` are called.

Likely targets:

- `Mmn._read_mmn`
- `Amn._read_amn`
- `Wannierize.write_hr`
- `Wannierize.write_tb`

Do not prioritize this before the iterative compute kernels unless profiling
shows that text I/O dominates.

### 8. Build a Numerical Validation Suite

Parallelization can change floating-point reduction order. Bitwise-identical
outputs should not be required unless explicitly needed.

Create validation cases that compare:

- final `omega_tot`,
- `OmegaI`,
- `OmegaD`,
- `OmegaOD`,
- Wannier centers,
- spreads,
- `_py_hr.dat`,
- `_py_tb.dat`.

Test both:

- standard symmetry mode: `-s`,
- site-symmetry mode: `-S`.

Use tolerances appropriate for floating-point changes. Track:

- absolute error,
- relative error,
- convergence iteration count,
- wall time,
- CPU utilization.

The representative Co calculation should be one benchmark case. Small example
cases from `examples/Cu` and `examples/Fe` can be used for quick correctness
checks, but they may be too small for meaningful performance conclusions.

## Recommended Implementation Order

1. Add detailed timers and collect baseline data.
2. Remove duplicate diagonalizations in `update_mmn` / `calc_expdw`.
3. Add explicit BLAS/MKL thread control.
4. Parallelize k-block `eigh` and `svd` operations.
5. Optimize `update_Mmn_by_Umat` with blocked or batched contractions.
6. Add diagnostic-output frequency controls.
7. Optimize text I/O and avoid repeated post-processing work.
8. Reconsider Ray or distributed execution only after single-node performance is
   well understood.

## Expected Outcome

Reasonable near-term target:

- 2x to 5x speedup on suitable calculations, depending on the actual bottleneck.

More ambitious target:

- approach Wannier90 wall time for some medium-sized workloads.

Hard target:

- consistently exceed Wannier90 performance.

The hard target likely requires more than Python-level scheduling. It may need
Numba, Cython, C++, Fortran kernels, or deeper use of batched LAPACK/BLAS.

## Notes for Future Ray Consideration

Ray may become useful later if:

- k-block tasks are coarse enough,
- large arrays can be shared without repeated copies,
- single-node shared-memory performance is already optimized,
- the target workload needs multiple nodes or many cores.

Ray should not be the first step because the current bottlenecks include
algorithmic duplication, BLAS thread policy, and memory layout issues that Ray
does not solve by itself.

## Implementation Progress on Co Benchmark

This section records the actual implementation and validation work performed on
the representative Co calculation.

### Benchmark Case

Original calculation:

```text
/home/user2/minkyu/computations/Co/pbesol-spn-kjpaw/wannier.sym
```

Benchmark copy:

```text
benchmarks/co_finite_iter
```

The benchmark uses symlinks to the original Co inputs and a copied `wannier.win`
with finite iteration counts:

```text
dis_num_iter = 5
num_iter     = 5
```

All performance runs were submitted through Slurm to `node03`; heavy
wannierization was not run on the login/master process.

Representative problem size:

```text
nk           = 1008
irreducible k-points = 76
nb           = 8
num_bands    = 52
num_wann     = 36
symmetry ops = 24
mode         = -S, site-symmetry
```

### Execution

The package entry point is:

```bash
symwannier wannierize -S wannier > stdout
```

For benchmark reproducibility, the Slurm script uses the equivalent module
entry point:

```bash
python -m symwannier.wannierize -S wannier > stdout
```

If an existing wrapper such as:

```bash
python $SymWann/wannierize.py -S wannier > stdout
```

points at the same `symwannier.wannierize` implementation, it uses the same
optimized code path. The important condition is that `PYTHONPATH` or the
installed package resolves to this updated repository.

The benchmark Slurm script is:

```text
benchmarks/co_finite_iter/job_benchmark.sh
```

It sets:

```bash
BLAS_THREADS=8
OMP_NUM_THREADS=$BLAS_THREADS
MKL_NUM_THREADS=$BLAS_THREADS
OPENBLAS_NUM_THREADS=$BLAS_THREADS
MKL_DYNAMIC=FALSE
```

`BLAS_THREADS=8` was selected after a thread sweep on `node03`.

### Performance Summary

Baseline source snapshot:

```text
tag: head
real time: 557.59s
internal total time: 9.26m
final omega_tot: 350.97317891
```

Current best:

```text
tag: diag_trial_matmul
real time: 48.61s
internal total time: 47.88s
final omega_tot: 350.97317891
```

Overall speedup on this finite-iteration Co benchmark:

```text
557.59s / 48.61s = about 11.5x
```

Current best major timers:

```text
mmn read                 :  3.50s
dis_extract              :  8.25s
wannierization           : 11.57s
post_process             : 16.50s
write_tb_file            : 11.15s
```

### Numerical Validation

Validation was done by comparing:

- final `omega_tot`,
- generated `wannier_py_hr.dat`,
- generated `wannier_py_tb.dat`.

For accepted optimizations, output differences were either exactly zero or
roundoff-sized. The current best accepted result matched the previous accepted
reference with:

```text
hr max diff = 0.0
tb max diff = 0.0
```

Some BLAS thread-count changes produced `tb` differences around `1e-9`, which
were treated as floating-point roundoff from changed reduction/order behavior.

### Implemented Optimizations

#### Time Reporting

`src/symwannier/timedata.py`

- Added human-readable duration formatting such as `2m 03.52s`.
- Replaced decimal-minute output such as `3.97m`.

#### Faster Full K-Point Lookup

`src/symwannier/sym.py`

- Added dictionary-based periodic k-point indexing.
- Relaxed the periodic k-point key precision from 8 to 6 decimal places.
- This removed expensive fallback linear scans in `search_ik_full`.

Observed effect:

```text
search_ik_full fallbacks: 4328 -> 0
Mmn full-k expansion: about 61.48s -> about 2.8s
mmn read: about 1m 02s -> about 3.5s
```

This was the largest single improvement.

#### Mmn Reading and Expansion

`src/symwannier/mmn.py`

- Replaced line-by-line `.immn` parsing with `np.fromstring`.
- Cached symmetry operation lookup results in `_mmn_full_klist`.
- Replaced per-block `einsum` matrix products with `np.matmul`.
- Added lightweight timing for raw parse and full-k expansion.

Accepted behavior:

```text
Mmn raw file read/parse: about 0.8s
Mmn full-k expansion:   about 2.8s
```

#### BLAS Thread Policy

`benchmarks/co_finite_iter/job_benchmark.sh`

- Added `BLAS_THREADS`, defaulting to `8`.
- Applies the value to `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and
  `OPENBLAS_NUM_THREADS`.

Thread sweep on `node03`:

```text
BLAS_THREADS=1   real 133.77s
BLAS_THREADS=2   real 148.33s
BLAS_THREADS=4   real 125.21s
BLAS_THREADS=8   real 124.25s
BLAS_THREADS=12  real 144.57s
BLAS_THREADS=24  real 142.00s
```

This was measured before the k-point key precision fix; after later
optimizations, `8` remained the benchmark default.

#### Batched Matrix Multiplication

`src/symwannier/wannierize.py`

Converted several `einsum` contractions to BLAS-backed `np.matmul`:

- `update_Mmn_by_Umat`
- `update_mmn0U`
- `calc_womegaI`
- `post_process` disentangled `Umat` composition
- `update_mmn` U-matrix updates
- exponent reconstruction from eigensystems

Observed effects included:

```text
update_Mmn_by_Umat: about 100s early in profiling -> about 6-7s after all changes
wannierization:     about 1m 55s baseline-era -> about 11.6s current best
```

#### Reused `calc_expdw` Eigensystems

`src/symwannier/wannierize.py`

- Diagonalizes `1j * dW` once per iteration.
- Reuses the eigensystem to build `exp(dW)`, `exp(dW/2)`, and
  `exp(alpha*dW)`.

This reduced repeated eigensolver work and simplified later `matmul` updates.

#### `calc_dw` Temporary Reduction

`src/symwannier/wannierize.py`

Rearranged the expression for the gradient-like `dW` calculation to combine
the `R` and `T` terms and reduce large temporary arrays:

- first combined `R + iT`,
- then summed over b-vectors before taking the anti-Hermitian part.

Observed effect:

```text
calc_dw: about 4.56s -> about 1.32s
```

#### `init_zmatrix` Optimization

`src/symwannier/wannierize.py`

- Replaced a large `einsum` with b-vector-wise batched `np.matmul` accumulation.

Observed effect:

```text
init_zmatrix: about 12.5s -> about 3s to 4s
```

#### Postprocess and Output

`src/symwannier/wannierize.py`

- Avoided repeated `calc_irvec`.
- Computed `ham_r` and Fourier factors once and passed them to both `write_hr`
  and `write_tb`.
- Used larger output buffers.
- Replaced high-frequency `.format()` calls in `write_hr` and `write_tb` with
  `%` formatting in hot loops.

Observed effect:

```text
write_tb_file: about 21.8s -> about 11.1s
post_process:  about 29s-37s -> about 16.5s
```

#### Trial Omega Diagonal Path

`src/symwannier/wannierize.py`

For the trial `alpha=1` and `alpha=0.5` omega evaluations inside `update_mmn`,
only diagonal `Mnn` is required. A first direct `einsum` implementation was
slower and was rejected. A second implementation using one batched `matmul`
plus a diagonal contraction was accepted.

Observed effect:

```text
wannierization: about 12.60s -> about 11.57s
```

### Rejected Attempts

The following approaches were tested and not kept:

- Python sparse `repmat` right-multiply in `Mmn._mmn_full_klist`
  - numerically correct, slower than dense BLAS.
- numba sparse `repmat` expansion
  - numerically correct, no practical speedup.
- postprocess cache stored as mutable object state
  - caused a `tb.dat` generation issue and was reverted.
- `write_tb` block `writelines`
  - slower after measurement.
- first direct `einsum` diagonal-only trial omega path
  - much slower than full `Mmn` update.

### Remaining Bottlenecks

At the current best point, the remaining expensive sections are:

```text
write_tb_file   about 11s
wannierization  about 12s
dis_extract     about 8s
```

The easiest Python/NumPy-level improvements have mostly been exhausted. Further
substantial improvement would likely require one of:

- compiled output writer for `.tb`,
- Cython/Numba/C++ kernels for repeated hot loops,
- deeper algorithmic changes to avoid generating full text `.tb` unless needed,
- broader benchmark validation across other materials and larger production
  iteration counts.

### Artifact Management

Large intermediate benchmark outputs named like:

```text
*_wannier_py_hr.dat
*_wannier_py_tb.dat
```

were deleted after validation to avoid consuming multiple gigabytes. The latest
accepted benchmark outputs remain as:

```text
benchmarks/co_finite_iter/wannier_py_hr.dat
benchmarks/co_finite_iter/wannier_py_tb.dat
```

Small logs, `stdout.*`, `stderr.*`, `time.*`, and Slurm logs were kept for
traceability.
