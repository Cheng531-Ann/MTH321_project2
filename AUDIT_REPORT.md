# Audit — Perpetual American Put LCP project

**Scope.** Does the code actually solve the problem as specified (Stages 1–2, LCP
formulation, free-boundary recovery, mesh study, `S_max` study, diagnostics)?
Where it does not, what exactly is wrong and what is the fix?

**Method.** I read every module, ran every script, and rebuilt an independent
reference (closed form for Stage 1, ODE shooting for Stage 2) in
`audit/reference_solver.py`. The shooting reference reproduces the Stage‑1
closed form to 10 significant digits — `S* = 71.4285714286`, `V(100) =
12.3200328678` — so it is trustworthy as a Stage‑2 reference too.
`audit/run_audit.py` regenerates every number quoted below; the raw output is
`audit/audit_output.txt`.

---

## 0. Verdict

The **core solver is sound**: every grid I ran returns a genuine discrete LCP
solution, with `min(a_h) ≈ -1e-11`, `min(b_h) = 0`,
`‖min(a_h,b_h)‖∞ ≈ 1e-11`, a single contiguous contact set, monotonicity,
convexity and `ψ ≤ V ≤ K` all satisfied. That part of the brief is met.

The **experimental design is not**, and two of the defects change the answer:

* **Stage 2 solves the wrong problem.** `q = 0.00`, not `0.02`.
* **Stage 1 poisons its own boundary condition** by imposing the analytic
  value at `S_max` instead of `V(S_max) = 0`, which also silently ignores
  `r`, `q`, `σ` and makes the `S_max` study vacuous.

Beyond those, the things the brief explicitly asks to *report* are either
missing (Stage 2 diagnostics), undocumented (free-boundary rule), or
hard-coded in the plotting layer rather than computed.

---

## 1. Code map

```
run_all.py                          orchestrator; runs 11 scripts, then checks 8 PNGs exist
├─ stage1_validation.py             ★ stencil + LCP solver + analytic solution + Σ study
├─ stage2_local_volatility.py       ★ same stencil with σ(S); N=3200 reference
├─ experiment_central_diff.py         central-difference variant (M-matrix demo)
├─ experiment_smax.py                 S_max sweep at fixed N=400
├─ experiment_solvers.py              policy iteration vs PSOR, N=400
├─ experiment_verification.py        LCP residual / shape checks (Stage 1 only)
├─ experiment_european_vs_american.py "European" linear solve vs American
├─ plot_lcp_verification.py           ★ copies the stencil again
├─ plot_solver_comparison.py          ← numbers are literals
├─ plot_smax_study.py                 ← numbers are literals
├─ plot_greeks.py                     ← S* = 74.25 is a literal
└─ __pycache__/perpetual_american_put_lcp.cpython-313.pyc   ← orphan (module gone)
```

★ = the discretisation is written out **six** times: `stage1_validation.py:105‑147`,
`stage2_local_volatility.py:66‑104`, `experiment_verification.py:31‑61`,
`experiment_solvers.py:39‑69`, `experiment_european_vs_american.py:34‑71`,
`plot_lcp_verification.py:33‑63`. They are identical apart from the Stage‑2
boundary term; any stencil fix has to be replicated six times, and `stage2`
already differs (it drops the right-hand-side term because `V_right = 0`).

---

## 2. Findings

### F1 — BLOCKING: Stage 2 uses `q = 0`, the brief specifies `q = 0.02`

`stage2_local_volatility.py:27`

```python
Q = 0.0            # should be 0.02
```

`main()` and `experiment_smax.py` both rely on the module default, so every
Stage‑2 number, plot and Greek in the repository is for a different problem.

| quantity | shipped (`q=0`) | specified (`q=0.02`) |
| --- | --- | --- |
| exact `S*` | 73.960493 | **69.166430** |
| exact `V(100)` | 13.248204 | **16.205476** |
| exact `V(400)` | 1.810793 | **3.383404** |

The error is ~3 in the option value and ~4.8 in the free boundary: larger than
anything else in the study.

**Fix:** `Q = 0.02`. Then recompute. Note `r - q = 0.03 > 0`, so the
backward-difference branch still applies.

---

### F2 — BLOCKING: Stage 1 imposes the analytic solution at `S_max`

`stage1_validation.py:130`

```python
V_right = float(exact_perpetual_put(np.array([S_max]), K)[0])
```

with (`stage1_validation.py:35‑46`)

```python
def exact_perpetual_put(S, K, S_star: float = 71.428571):
    V[~exercise] = (K - S_star) * (S[~exercise] / S_star) ** -2.5
```

Three distinct problems:

1. **It uses the answer as a boundary condition.** The comment says this
   "avoids an artificial truncation boundary layer" — but removing truncation
   error is exactly what the `S_max/K = 3,4,6` study is supposed to *measure*.
   As a result the `S_max` study (F4) has nothing to measure.
2. **The brief says `V(S_max) = 0`.** `V(400) = 0.385` is not small compared
   with the `1e-3`…`1e-4` discretisation errors being reported.
3. **It silently ignores `r`, `q`, `σ`.** The function hard-codes `λ = -2.5`
   and `S* = 71.428571`, which are only correct for `r = 0.05`, `q = 0`,
   `σ = 0.2`. `solve_perpetual_american_put` nevertheless accepts `q` and
   `sigma` and feeds them to the stencil, so the operator and the boundary
   describe different problems.

Evidence (S13 of the audit) — passing `q = 0.02`, which the signature invites:

```
correct  λ = -1.850781,  S* = 64.921894,  V(400) = 1.212082
exact_perpetual_put(400)                = 0.385001        ← q = 0 answer
project:   S_h* = 65.000 (true 64.922),  max|V_h − V_true| = 8.271e-01
```

An 0.83 error where the method should give `1e-3`. The validation loop
(`main`) is equally hard-wired: it prints `Exact S* = 71.428571` regardless of
the parameters it just used.

**Fix:** make the analytic solution parameter-aware, and use `V(S_max) = 0`
everywhere.

```python
def exact_perpetual_put(S, K, r=0.05, q=0.0, sigma=0.2):
    a = 0.5 * sigma**2
    b = (r - q) - a
    lam = (-b - math.sqrt(b*b + 4*a*r)) / (2*a)      # decaying root
    S_star = lam * K / (lam - 1.0)
    ...
```

---

### F3 — MAJOR: `S_max = 400` is far too small once `V(S_max) = 0` is used

Fixed `h = 0.25`, Stage 1 (S4). `zeroBC` is the honest boundary condition:

| `S_max` | `N` | error, exact BC (project) | error, `V(S_max)=0` | `V(100)` |
| --- | --- | --- | --- | --- |
| 200 | 800 | 1.25e-04 | **2.178** | 11.5519 |
| 300 | 1200 | 1.25e-04 | **0.790** | 12.1368 |
| 400 | 1600 | 1.25e-04 | **0.385** | 12.2534 |
| 600 | 2400 | 1.25e-04 | 0.140 | 12.3039 |
| 1200 | 4800 | 1.25e-04 | 0.0247 | 12.3186 |

exact `V(100) = 12.3200328678`.

So at the required `S_max/K = 3, 4, 6` the truncation error is **0.79 / 0.385 /
0.140** — two to four orders of magnitude *above* the discretisation error the
project reports. The `h`-study and the `S_max`-study cannot both use
`S_max = 400`.

Same for Stage 2, `q = 0.02` (S8, fixed `h = 0.25`):

| `S_max` | 200 | 300 | 400 | 600 | 1200 | 2400 |
| --- | --- | --- | --- | --- | --- | --- |
| error in `V(100)` | −2.03 | −0.754 | **−0.379** | −0.145 | −0.0284 | −0.0056 |
| error at `S_max` | 7.20 | 4.63 | 3.38 | 2.18 | 1.03 | 0.483 |

(The second row is exactly true `V(S_max)` — the boundary value that was
thrown away.)

**Fix:** either use `S_max ≈ 20K` (2400) for the `h`-study, or apply a
Robin/asymptotic far-field condition, and report the `S_max` study as a
separate table at fixed `h`.

---

### F4 — MAJOR: the `S_max` study changes `h`, so it measures nothing

`experiment_smax.py:10‑11`

```python
N = 400
S_MAX_VALUES = (200.0, 300.0, 400.0, 500.0)
```

`h = S_max / 400 = 0.5, 0.75, 1.0, 1.25`. Mesh error and truncation error move
together — precisely what the brief forbids ("repeat … with `S_max/K = 3,4,6`
so that truncation error is not confused with mesh error"). The values in
`(200, 300, 400, 500)` are `2K, 3K, 4K, 5K`, not `3K, 4K, 6K`.

Measured (S9, `q = 0` as shipped): `V(100) = 11.707, 12.684, 12.972, 13.157`.
The `0.976` jump from `S_max = 200 → 300` is a mesh artefact, not truncation.

**Fix:** hold `h` fixed (`N = round(S_max / h)`) and vary `S_max`, as in S8.

---

### F5 — MAJOR: free-boundary extraction is a bare node snap with an undocumented absolute threshold

`stage1_validation.py:221‑226` (identical at `stage2_local_volatility.py:167‑172`)

```python
above_obstacle = np.flatnonzero(V > psi + 1e-6)
first_above = int(above_obstacle[0])
S_free = float(S[first_above - 1])
```

* **No interpolation** — the estimate is `S_{i₀}`, the last contact node, so
  the error is bounded by `h` and is one-sided (always `≤ S*`).
* **No stated tolerance** — `1e-6` is absolute on a quantity of size `K = 100`.
  It happens to work here because `b = V − ψ ≈ ½V''(S*)(S−S*)²` and
  `½V''(S*) ≈ 0.0245`, so the threshold only fires for `S − S* > 0.0064`;
  that is luck, not design.
* **The resulting sequence does not converge** — measured (S1, S3):

| `N` | 100 | 200 | 400 | 800 | 1600 |
| --- | --- | --- | --- | --- | --- |
| `S_h*` | 72.0 | 72.0 | 71.0 | 71.5 | 71.5 |
| `|S_h* − S*|` | 0.571 | 0.571 | 0.429 | 0.071 | 0.071 |
| order | — | 0.00 | 0.41 | 2.59 | 0.00 |

  Four refinements that do not decrease. No convergence rate can be quoted
  from this.

Stage 2 is worse: `S_h* = 76, 74, 74, 74` for `N = 100…800` (S6) — completely
static over three refinements, then the `N = 3200` reference reports `74.25`,
which `plot_greeks.py:20` then hard-codes as truth.

**Fix:** state the rule, bracket `S*`, and stop snapping. Minimum acceptable
version:

```python
i0 = last index with b_i <= tol_slack           # tol_slack = 1e-10 * K
S_lo, S_hi = S[i0], S[i0 + 1]                   # S* is in [S_lo, S_hi]
S_h = 0.5 * (S_lo + S_hi)                       # |S_h - S*| <= h/2
```

Better: fit `sqrt(b) = α + β S` over the first few continuation nodes
(`b ~ c(S−S*)²` is forced by smooth fit) and take `S* = −α/β`. I measured that
this reduces the constant by 2–4× but **does not** restore second order
(S3: `4.59, 2.10, 0.816, 0.505, 0.242, 0.0998`, order ≈ 1.2), because the
discrete solution itself only resolves the kink to `O(h)`. Report
`|S_h* − S*| = O(h)` honestly, with the bracket.

---

### F6 — MAJOR: Stage 2 reports none of the required diagnostics

The brief requires, for Stage 2, `min_i(a)`, `min_i(b)`, `‖min(a,b)‖∞`,
`ψ ≤ V ≤ K`, monotonicity, convexity, a single contiguous contact set, the
initial iterate, the iteration count and the stopping rule.

`solve_local_volatility_put` returns `max_error = nan`
(`stage2_local_volatility.py:179‑180`), performs **no** shape or residual check
(contrast `stage1_validation.py:216‑219`, which at least asserts the
M‑matrix property), and `main()` prints only `S_h*`, an error and an iteration
count. None of the LCP quantities appear anywhere for Stage 2.

Measured for reference (S10, `q = 0.02`, `V(S_max)=0`, `S_max=400`), so this is
cheap to add:

| `N` | `min a` | `min b` | `‖min‖∞` | monotone | convex | 1 contact set |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | −2.27e-13 | 0 | 2.27e-13 | yes | yes | yes |
| 400 | −3.64e-12 | 0 | 3.64e-12 | yes | yes | yes |
| 1600 | −5.82e-11 | 0 | 5.82e-11 | yes | yes | yes |

Also: `experiment_verification.py` checks `ψ ≤ V ≤ K`, monotonicity and
convexity for Stage 1 only, and its convexity test is `np.diff(V, 2) >= -1e-6`
— an absolute tolerance on a second difference whose scale is `h²V''`. Passes
here, but it is not a scale-free test.

---

### F7 — MAJOR: two figures are drawn from literal arrays, not from the computation

`plot_smax_study.py:10‑12`

```python
S_max  = np.array([200.0, 300.0, 400.0, 500.0])
V_100  = np.array([11.707, 12.683, 12.971, 13.093])
asymptotic_value = 13.093
```

`plot_solver_comparison.py:11‑12`

```python
iterations = np.array([102, 67_758], dtype=float)
cpu_times  = np.array([0.016, 18.42], dtype=float)
```

Consequences:

* Neither figure is regenerated by `run_all.py`; the script only checks that
  the PNG *exists* (`run_all.py:58`), so stale figures pass the "all
  experiments generated successfully" check forever.
* The CPU times disagree with the actual run. Measured just now:
  **policy 0.0601 s, PSOR 25.08 s**; the figure shows **0.016 s / 18.42 s**.
* `asymptotic_value = 13.093` is not a limit, it is the `S_max = 500` sample.
  It is labelled "Asymptotic limit".
* The annotation "Beyond `S_max = 300`, the truncation error is negligible" is
  false: the sequence is still moving by `0.124` from `400 → 500` and is
  `≈0.1` away from `V(100)` at `S_max = 10⁴`.

**Fix:** have the plotting scripts import and call the experiment functions
(they are already importable), or write the sweep to a CSV/JSON that the plot
reads.

---

### F8 — MAJOR: the drift difference is taken on the wrong side, so the stencil is *anti*-upwind

`stage1_validation.py:91‑120` uses the second-order **backward** difference

```
V'_i ≈ (3V_i − 4V_{i−1} + V_{i−2}) / (2h)
```

which contributes `+2·drift` to the `V_{i−1}` coefficient of `A = −L_h`:

```python
lower1_full = -diffusion + 2.0 * drift
```

For `b = (r−q)S > 0` this coefficient is **positive** whenever
`2b/h > a/h²`, i.e. `S < 4(r−q)h/σ²`. A positive off-diagonal is exactly what
the M-matrix property forbids. The monotone first-order stencil for `A = −L` is
on the `V_{i+1}` side:

Measured (S2) — count of positive off-diagonals in the raw stencil:

| scheme | `N=100` | `N=200` | `N=400` | `N=800` | `N=1600` |
| --- | --- | --- | --- | --- | --- |
| project (BDF2 backward + limiter) | 4 | 4 | 4 | 4 | 4 |
| forward `V_{i+1}` difference | 0 | 0 | 0 | 0 | 0 |
| fitted (Scharfetter–Gummel) | 0 | 0 | 0 | 0 | 0 |

So `A` is an M-matrix only *because* of the limiter at
`stage1_validation.py:126‑128`, which absorbs the offending coefficient into the
diagonal:

```python
non_monotone = lower1_full > 0.0
diag_full[non_monotone] += lower1_full[non_monotone]
lower1_full[non_monotone] = 0.0
```

Two things to say about that repair:

* It is a **first-order, non-consistent** modification (it changes the row by an
  `O(1)` amount in the `S = O(h)` band), and its stated justification — "Near
  `S = 0`" — understates the band: it is `S < 4(r−q)h/σ²`, which is `5h`
  (four rows) at `σ = 0.2`, but grows like `1/σ²`.
* It happens to be **harmless for the default parameters** because
  `S* = 71.4 ≫ 5h`, so the patched rows always sit inside the contact region
  where `V = ψ` is clamped. Nothing in the code checks that, and the safety
  margin evaporates as `σ` falls. Measured with everything else fixed
  (`N = 800`, `S_max = 400`, exact far-field value):

  | `σ` | `S*` | limiter band | proj error | fitted error |
  | --- | --- | --- | --- | --- |
  | 0.20 | 71.43 | `S < 2.5` | 1.30e-03 | 1.25e-04 |
  | 0.08 | 93.99 | `S < 15.6` | 6.22e-03 | 8.69e-05 |
  | 0.04 | 98.43 | `S < 62.5` | 2.31e-02 | 1.78e-03 |
  | 0.02 | 99.60 | `S < 250.0` | **9.96** | 3.96e-03 |

  At `σ = 0.02` the band covers the whole grid; the modified stencil is applied
  through the entire continuation region and the answer is wrong by `O(1)`.
  The `M`-matrix assertion at `stage1_validation.py:216‑219` still passes,
  because the limiter *makes* it pass. That is the tell: the check validates
  the patch, not the scheme.
* It is *worse* than plain central differencing: `experiment_central_diff.py`
  reports **1** positive off-diagonal at `N = 400` for central differences, and
  uses that to argue upwinding is necessary — while the "upwind" scheme it is
  being compared against has **4**. The narrative of that experiment does not
  support its conclusion.

**Fix:** replace both the BDF2 stencil and the limiter with exponential fitting.
It is a true M-matrix at every cell Péclet number, needs no patch, and is
measurably more accurate:

```python
pe = b * h / a
bp = pe / np.expm1(pe)          # B(Pe)
bm = (-pe) / np.expm1(-pe)      # B(-Pe)
lower1 = -(a / h**2) * bp
upper  = -(a / h**2) * bm
diag   = (a / h**2) * (bp + bm) + r
```

Measured on Stage 1 (S2), same grid, same solver:

| `N` | project | fitted | forward-1 |
| --- | --- | --- | --- |
| 200 | 1.80e-02 | **7.91e-03** | 2.28e-01 |
| 400 | 4.46e-03 | **4.30e-03** | 1.18e-01 |
| 800 | 1.30e-03 | **1.25e-04** | 5.85e-02 |
| 1600 | 2.84e-04 | **1.25e-04** | 2.93e-02 |

`audit/reference_solver.py` implements all three (`scheme="proj" | "fwd1" |
"fitted"`).

---

### F9 — MEDIUM: the "policy iteration" is a one-node-per-sweep active-set iteration

`stage1_validation.py:161‑207`. The update only ever *releases* rows

```python
active_int[active_int & (lcp_residual < -1e-10)] = False
```

and re-derives the rest from `V <= psi + 1e-10`. That is not Howard's rule
(which would be `new_active = a > b`, applied to every row every sweep). The
contact front therefore advances by roughly one node group per sweep and the
iteration count is `Θ(N)`:

| `N` | 100 | 200 | 400 | 800 | 1600 |
| --- | --- | --- | --- | --- | --- |
| project iterations | 26 | 51 | 102 | 201 | 401 |
| Howard iterations | 8 | 15 | 30 | 58 | 115 |

It converges to the right answer — the LCP residuals in F6/S1 confirm that —
but calling it "policy iteration" and benchmarking it against PSOR
(`plot_solver_comparison.py`) overstates what the comparison shows: one
linearly-converging sweep scheme against another. (PSOR itself: 67 758 sweeps,
25.1 s at `N = 400`, ending `4.4e-7` away from the policy solution, i.e. its
stopping rule `max|Δx| < 1e-10` is not the LCP residual and it stops
under-converged relative to the tolerance its table claims.)

The brief only asks for PSOR *or* policy iteration plus a stated initial
iterate / relaxation / count / stopping rule, so this is not a correctness
failure — but the initial iterate (`V = ψ`), the stopping rule
(`max|ΔV| < 1e-12` **and** a stable policy) and the fact that the count scales
with `N` should be stated, not implied.

---

### F10 — MEDIUM: the European comparison is not a European option

`experiment_european_vs_american.py`. A perpetual European put is identically
zero: `V_E = lim_{T→∞} E^Q[e^{−rT}(K − S_T)^+] ≤ lim K e^{−rT} = 0`. The module
instead solves the linear problem with `V(0) = K` and
`V(S_max) = V_exact(S_max)`. But the only bounded solution of
`½σ²S²V'' + (r−q)SV' − rV = 0` near `S = 0` is the `S¹` branch, which vanishes
at the origin, so `V(0) = K` is incompatible with it. The solver then produces a
numerical boundary layer of width `O(h)` at `S = 0`.

Measured output:

```
Maximum early exercise premium  = 94.42086225   at S = 4.000
Premium for S >= S*            in [0.000000e+00, 2.842993e+01]
```

A premium of 94.4 at `S = 4` means the "European" value there is ≈ 1.6, whereas
the harmonic branch fitted at `S_max` gives ≈ 0.004 — a factor 400, i.e. the
layer. Since the true `V_E ≡ 0`, the plotted "premium" is just `V_American`, and
the figure and its caption ("the premium remains positive beyond the American
exercise boundary") convey nothing.

**Fix:** drop the experiment, or replace it with something well posed (e.g. the
same LCP with the obstacle removed but with `V(0) = K − 0` replaced by the
natural/Neumann condition, stated explicitly).

---

### F11 — MINOR but user-visible

* `plot_greeks.py:20` — `S_star = 74.25` instead of `result.S_free`. And `74.25`
  is itself a node-snapped Stage‑2 reference (`h = 0.125`), not the true
  73.960493.
* `run_all.py:58` — checks only that figures exist, never that they are newer
  than the scripts. Any hard-coded figure passes.
* `__pycache__/perpetual_american_put_lcp.cpython-313.pyc` — orphan for a
  module that no longer exists. Delete it; it will confuse anyone reading the
  directory.
* `LCPResult.max_error` is `float("nan")` for Stage 2, and the Stage‑1 value is
  a full-grid sup norm that includes the boundary node where the exact value is
  imposed (so it is flattered by exactly one node).
* `observed_order` divides by `log 2` assuming a refinement ratio of 2 — fine
  here, undocumented.
* `stage2_local_volatility.py:210‑212` indexes the reference with
  `reference.V[::stride]`, which silently requires `REFERENCE_N % N == 0`.
* Stage 2's error is `max|V_h − V_ref|` where the reference is itself truncated
  at the same `S_max`. Two grids sharing a wrong boundary condition converge to
  each other, so this measures `h`-convergence only; it cannot detect F1/F3.

---

### F12 — the "p ≈ 2.0" claim is not supported by the reported numbers

`stage1_validation.py:314‑318` prints

> "The second-order BDF2 drift gives p approximately 2.0. The N=800 value is
> slightly lower because the free boundary is not aligned with the grid…"

Measured `p = 2.169, 2.014, 1.775, 2.196` — non-monotone, and the explanation
offered is a guess. The sup norm is not attained at a stable location either:

| `N` | 100 | 200 | 400 | 800 | 1600 |
| --- | --- | --- | --- | --- | --- |
| argmax node `S` | 100 | 100 | 72 | 97 | 99.75 |

and the signed error changes sign just above `S*` (`−4.46e-3` at `S = 72`,
`+3.2e-4` at `S = 80` for `N = 400`). So the sup norm mixes the
free-boundary-position error (dominant at coarse `N`) with a broad smooth hump;
it is a single fragile functional.

The clean way to see how much of it is grid-alignment luck is the fitted
scheme, where the peak is always at the first continuation node and equals
`½V''(S*)(S_i − S*)²`:

| `N` | `h` | first continuation node | `δ = S_i − S*` | sup error |
| --- | --- | --- | --- | --- |
| 100 | 4 | 72 | 0.571 | 7.905e-03 |
| 200 | 2 | 72 | 0.571 | 7.905e-03 |
| 800 | 0.5 | 71.5 | 0.071 | 1.248e-04 |
| 1600 | 0.25 | 71.5 | 0.071 | 1.248e-04 |

Identical to five digits for `N = 100/200` and for `N = 800/1600`, because in
each pair the node nearest `S*` is the same physical point. A convergence table
built only on the grid sup norm can therefore show `p = 0` or `p = 5` on
consecutive refinements for reasons that have nothing to do with the scheme.

---

## 3. What is correct

Worth stating plainly, because it is most of the numerical difficulty:

* The LCP is formulated correctly and the contact region is imposed as a
  region (`V = ψ` on active rows), not as a condition at `S*` alone.
* The discretisation *is* consistent and second-order away from
  `S = O(h)`; the assembly offsets `(−2, −1, 0, 1)` and the boundary
  bookkeeping (`b_boundary[0]`, `b_boundary[1]`, `b_boundary[-1]`) are right.
* The solver reaches a genuine LCP solution on every grid:
  `min(a_h) ≈ −1e-12 … −6e-11`, `min(b_h) = 0`, `‖min‖∞ ≈ 1e-12 … 1e-10`
  (the residual grows with `N` because the release tolerance `1e-10` is
  absolute) — comfortably better than the release tolerance it is allowed.
* Contact set is always contiguous from `S = 0`; `ψ ≤ V ≤ K`, monotone
  decrease and convexity all hold for Stage 1 and (measured, S10) for Stage 2.
* Stage 1 achieves `2.8e-4` sup error at `N = 1600`, which is respectable.
* The `N = 3200` reference and the `h`-refinement structure in
  `stage2_local_volatility.py` are reasonable; only the `S_max` side is missing.

---

## 4. Minimal patch list

1. `stage2_local_volatility.py:27` — `Q = 0.02`.
2. `stage1_validation.py:35‑46, 130` — parameter-aware analytic solution; use
   `V(S_max) = 0` in both stages (or a stated asymptotic condition).
3. Both solvers — add a `V_right`/`right_bc` argument so the same code can do
   exact-BC (validation) and zero-BC (production), and print which was used.
4. `S_max ≈ 20K` for the `h`-study; `S_max/K ∈ {3,4,6}` for the `S_max` study
   **at fixed `h`**.
5. Free boundary — bracketed, documented tolerance, interpolation; return
   `S_lo, S_hi, S_h` and print the bracket.
6. Stage 2 — add the `min a / min b / ‖min‖∞ / ψ≤V≤K / monotone / convex /
   contiguous` block and the initial iterate + stopping rule to the printed
   table.
7. `plot_smax_study.py`, `plot_solver_comparison.py` — import the experiments
   instead of literal arrays; make `run_all.py` compare mtimes.
8. `plot_greeks.py` — use `result.S_free`.
9. Replace the BDF2-backward + limiter pair with the fitted stencil (F8) and
   fold the six copies into one module.
10. Drop `experiment_european_vs_american.py` or restate the problem.

---

## 5. What the submission still has to contain

Cross-checking the brief item by item, against what the code currently emits:

| required by the brief | Stage 1 | Stage 2 |
| --- | --- | --- |
| `min_i(a_{h,i})` | computed, `experiment_verification.py` | **missing** |
| `min_i(b_{h,i})` | same | **missing** |
| `‖min(a_h,b_h)‖∞` | same | **missing** |
| `ψ_h ≤ V_h ≤ K` | yes | **missing** |
| monotone decrease | yes | **missing** |
| convexity | yes (absolute tol) | **missing** |
| single contiguous contact set from `S=0` | **not checked anywhere** | **missing** |
| initial iterate / relaxation / iteration count / stopping rule | in code, not in the report | partially |
| explicit `S_h*` rule + tolerance + interpolation | **no** | **no** |
| ≥ 4 refinements | 4 (`N=100..800`) | 4 (`N=100..800`) |
| `‖V_h − V‖∞` and `|S_h* − S*|` tables | value only, `|ΔS*|` only printed | no `|ΔS*|` vs reference |
| refined-grid reference | n/a (closed form) | `N=3200`, but same `S_max` |
| `S_max/K ∈ {3,4,6}`, decoupled from `h` | **no** | **no** |

---

## 6. Files added by this audit

| file | purpose |
| --- | --- |
| `audit/AUDIT_REPORT.md` | this report |
| `audit/reference_solver.py` | independent solver: three stencils, Howard policy iteration, free-boundary extraction, closed form, ODE shooting reference |
| `audit/run_audit.py` | rebuilds every table above (S1–S14) |
| `audit/audit_output.txt` | captured output of the above |

Run it with `python audit/run_audit.py` from the project root (about 2 minutes).
Nothing in `audit/` imports the project's stencil — only its two public solvers
— so agreement is evidence, not shared code.

The shooting reference is itself validated: for constant `σ = 0.2`, `q = 0` it
returns `S* = 71.4285714286` and `V(100) = 12.3200328678`, identical to the
closed form.
