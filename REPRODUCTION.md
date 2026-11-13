# CL-MEE revised experiment protocol and reproduction guide

## 1. Reproduction modes

The recommended reference reproduction uses the validation-trained scorer
checkpoint `foursquare_model_weights.json`. This makes the downstream POI
ranking and the manuscript's reference metrics reproducible without depending
on small platform-specific differences in the SciPy optimizer.

From this directory, on Windows CMD or PowerShell:

```cmd
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe run_revision_experiments.py --dataset dataset_TSMC2014_NYC.txt --output-dir verification_results
.venv\Scripts\python.exe verify_package.py --results-dir verification_results --dataset dataset_TSMC2014_NYC.txt --paper ..\main.tex --reference-dir revision_outputs --pdf ..\main.pdf --max-pages 9
```

Alternatively, double-click `RUN_ALL_TESTS_WINDOWS.bat` in this folder. It
creates/uses `.venv`, installs pinned dependencies, captures the environment,
runs the tests and full experiments, and checks the generated outputs against
the supplied reference results and manuscript.

The default runner loads the supplied validation-fitted weight checkpoint.
To independently rerun the weight optimizer instead, use:

```cmd
.venv\Scripts\python.exe run_revision_experiments.py --dataset dataset_TSMC2014_NYC.txt --output-dir refit_results --refit-weights
```

`--refit-weights` is a diagnostic/refitting mode. Floating-point optimization
can differ slightly across OS, BLAS, and SciPy builds; therefore, the default
checkpoint mode is the reference mode for reproducing the reported paper
metrics. The checkpoint includes its feature order, dataset hash, training
sample metadata, fitted weights, and source environment.

## 2. Dataset and reference environment

- Dataset: `dataset_TSMC2014_NYC.txt` (included in this full archive).
- SHA-256: `2f39ce698e5bff6683c74bec30c436cc022c160b0f3d73c5268b55491b2445f9`.
- Reference Windows environment supplied by the author: CPython 3.13.15,
  64-bit AMD64; NumPy 2.3.5; pandas 2.2.3; SciPy 1.17.0; scikit-learn 1.8.0.
- Pinned dependencies are listed in `requirements.txt`.
- `capture_environment.py` writes the actual platform details and `pip freeze`
  for the machine on which the reproduction is run.

Dataset statistics: 227,428 check-ins, 1,083 users, and 38,333 venues;
34,117 training POIs; 20,211 eligible validation transitions and 20,223
eligible test transitions. Test targets are restricted to venues observed in
training.

The reference scorer weights are:

- User preference: 0.3062301385743233
- Proximity: 0.2233927855515908
- Popularity: 0.4085081815201605
- Temporal compatibility: 0.022742130750884118
- Transition compatibility: 0.039126763603041244

They were fitted on 5,000 eligible validation transitions, using 20 sampled
negatives per transition, pairwise logistic loss, L2 coefficient 0.1,
non-negative bounds, and sum-to-one normalization. The exact fitted vector is
stored in `foursquare_model_weights.json`; `--refit-weights` repeats fitting.

## 3. Foursquare NYC protocol

1. Sort check-ins by `(user_id, utc_time, venue_id)` using a stable sort.
2. Split each user's chronologically ordered history 80/10/10 using the
   within-user position fraction `(position + 1) / history_length`.
3. Construct five pre-visit features:
   - User-category preference: category counts with 0.5 additive smoothing,
     normalized by each user's maximum category value.
   - Proximity: `exp(-distance_km / tau)`, where `tau` is the median movement
     distance in training transitions (1.5114327113609778 km in the reference run).
   - Popularity: `log1p(training_count) / log1p(max_training_count)`.
   - Temporal compatibility: six four-hour local-time bins, 0.5 smoothing,
     column-normalized and scaled by the maximum for each category.
   - Category-transition compatibility: 0.5-smoothed previous/current-category
     transition matrix, row-normalized and scaled by each row maximum.
4. Fit non-negative scorer weights on a seeded sample of 5,000 validation
   transitions (`random_state=42`). For each event, sample up to 10 same-category
   negatives, add nearby negatives until there are 15, then use popularity
   sampling to reach 20 negatives. Optimize pairwise logistic loss plus L2
   regularization (`alpha=0.1`) with non-negative bounds; normalize weights to
   sum to one. The fitted coefficients are distributed as a checkpoint.
5. Build each test candidate set from the true target, 249 nearest training POIs,
   and 250 popularity-focused POIs sampled from the 5,000 most popular training
   POIs. Candidate construction uses seed `20260919`.
6. Rank 500 candidates and apply CL-MEE only to the top 40. Score ties preserve
   candidate order; popularity ties are resolved by venue ID. Equal timestamps
   for the same user are ordered by venue ID.

## 4. Reviewer-requested additions

### Budget-matched post-hoc baseline

The baseline preserves the RankOnly ordering and applies the same positive
evidence-prefix calculation and hard truncation for budget B. It does not
promote any candidate. This isolates the effect of display truncation from
explanation-aware re-ranking.

### Comparison-reference sensitivity

The primary method retains the submitted hybrid anchor: the incumbent is
compared with the runner-up and challengers are compared with the incumbent.
The sensitivity check compares every top-M candidate to the common runner-up
anchor. It is a sensitivity analysis, not a replacement of the primary method.

### Component ablations

- Budget ablation: vary B=1,2,3 while fixing lambda=0.03, epsilon=0.04, and
  eta=0.008.
- No-over-budget-penalty ablation: set lambda=0, hold epsilon=0.04 and
  eta=0.008, and vary B.
- Truncation-only baseline: preserve the base ranking and apply hard display
  truncation for each B.
- P3 is an explicit ranking-identity control and bypasses adjusted-score
  maximization; it is not treated as an ordinary parameter setting.

### Operational diagnostics export

Each run writes `operational_diagnostics.csv` with separate all-event and promoted-event score-loss distributions for P1 and P2, promotion-depth quantiles, pre-truncation over-budget rates, below-$\rho$ coverage rates, and invariant counts. The JSON remains the full machine-readable record.

### Evidence and zero-evidence convention

For D=0, K=m is only a conservative ranking-burden surrogate; no positive
reason exists and displayed units L=0. Budget exceedance and below-rho coverage
rates are therefore reported both over all events and conditional on positive
D. Hard truncation guarantees L<=B; it does not guarantee F_model>=rho.

### Statistical aggregation

Overall Foursquare Recall@5, NDCG@5, and MRR are event-weighted means over the
20,223 eligible test transitions. For user-level paired inference, event NDCG
contributions are first averaged within each user and the resulting user means
are then equally weighted over 1,083 users. These are different estimands and
need not have the same numerical mean. Simulation paired tests are performed
across the ten fixed seeds.

### Runtime

The latest author-provided Windows CMD timing run (CPython 3.13.15, AMD64)
reported candidate generation, feature scoring/top-M, and evidence/policy
evaluation times of 8.762 s, 1.902 s, and 0.762 s, respectively. The latter is
0.0377 ms per event and 40.09% of the measured feature-scoring stage. These are
single-run wall-clock measurements; the supplied CMD transcript did not record
the CPU model or RAM, so they are not general latency guarantees. Each new run
records its own environment and timing; rerun on target hardware before making
deployment claims.

## 5. Authoritative reference results (Windows checkpoint run)

The authoritative machine-readable reference is in `revision_outputs/`.
The user-supplied Windows run used Python 3.13.15 and the pinned libraries.

- Simulation: 24,000 sessions; CL-MEE NDCG@5 = 0.1306087614 (99.68% of
  RankOnly 0.1310307307); 1.18225 displayed units; mean F_model = 0.9719348.
- Simulation budget-matched post-hoc: 1.175958 displayed units and mean
  F_model = 0.9505293. Paired seed-level p-values: NDCG difference p=0.08514,
  units difference p=0.0003137, and F_model difference p=2.128e-9.
- Foursquare RankOnly NDCG@5 = 0.1565492882.
- Foursquare P1 NDCG@5 = 0.1565238900 (99.9837763% retained; report 99.984%).
- Foursquare P2 NDCG@5 = 0.1567317884 (100.1165769% retained; report 100.117%).
- P1/P2 display-budget violations after truncation = 0 by construction.
- P1 user-level paired t-test p=0.49565 and Wilcoxon p=0.21445; P2 p=0.55569
  and 0.49543. Non-significance is not evidence of equivalence.

The JSON contains full operational diagnostics, score-loss and promotion-depth
distributions, user-level statistics, anchor sensitivity, and runtime details.
`operational_diagnostics.csv` exports the all-event and promoted-event P1/P2
distributions in a compact two-row table. The CSV files are exports of the
corresponding JSON tables. `verify_package.py` checks their consistency,
verifies simulation and Foursquare manuscript metrics, checks the runtime
reference, and can check the compiled PDF page count.

No user study was conducted. The results do not establish lower cognitive load,
improved comprehension, greater trust, satisfaction, or field effectiveness.
