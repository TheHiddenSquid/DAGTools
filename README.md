# DAGTools · DAG Lab

A local interactive workbench for the node-colored Gaussian DAG algorithms in
[DAGdiscovery](https://github.com/TheHiddenSquid/DAGdiscovery). Draw a generating
model, generate or import observations, explore a candidate DAG with a live BIC
score, and run the original discovery implementations or Gaussian GES.

The original DAGdiscovery repository is unchanged. In the separate DAGTools
repository, the `original-colorednodes` tag preserves the research implementation
before the application changes.

## Launch

On macOS, double-click **Start DAG Lab.command**. The application opens at
**http://127.0.0.1:8765**. Keep its Terminal window open; Control-C stops the
server and its active algorithm worker. The first launch creates an isolated
Python environment and builds the interface if necessary.

Requirements: Python 3.11 or newer (tested on Python 3.12) and Node.js 20 or newer
for building the interface. Initial setup downloads dependencies; subsequent
use works offline. The launcher also installs updated Python requirements when their
files change. The Python dependencies are pinned in `requirements-lock.txt`
and the frontend dependencies in `frontend/package-lock.json`.

For manual setup:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install --no-deps -e .
cd frontend
npm ci
npm run build
cd ..
.venv/bin/python -m daglab.cli
```

Use `--port 8766` to choose another port, `--no-browser` to skip opening a tab,
or `--data-dir /path/to/data` to choose a storage location.

## Workflow

1. **Create data.** Edit the generating DAG. Select nodes to rename them, assign
   variance groups, and set each group's noise variance. Select edges to edit
   coefficients or reverse/delete them. Nodes are circles shaded by variance
   group; straight arrows meet their boundaries in any direction. Add edges by dragging between ports or
   using the selected node's **Connect to** control. Generate a sample with an
   explicit size and seed, or import a UTF-8 CSV with an optional header row.
2. **Explore & score.** Edit a separate candidate graph on the active sample's
   variables. BIC updates when its edges, partition, or centering change.
   **Optimize partition** keeps the graph and finds the best variance partition
   using the existing dynamic-programming routine. **Copy reference** restores
   the graph and partition used to generate that sample.
3. **Run algorithms.** Choose MCMC, greedy graph-and-color search, greedy
   search with an optimal partition, or GES with Gaussian BIC. Settings appear from the Python registry.
   A separate worker executes a frozen snapshot of the inputs. You can keep
   editing while it runs. Starting a run saves the current experiment and attaches
   the run to its own history; reopening that experiment restores its history.
   Inspect the trace, DAG, or partially directed result,
   then choose **Send to Explore & score** to load the result into the candidate
   editor. **Use as next starting DAG** loads the same candidate while keeping
   you on the algorithm page: choose another algorithm and run it. Both actions
   select (or restore) the result's sample and enable **Use DAG from Explore &
   score**. The starting-DAG summary shows the current graph's size. MCMC starts
   from that graph; greedy searches use it for their first restart.
4. **Save.** Save named experiments locally. Export experiment JSON includes
   the observations, both graphs, and settings; it can be imported on another
   computer. Run JSON preserves the run inputs, diagnostics, score trace,
   dataset fingerprint, code revision, and output. Exact MCMC score history can
   optionally be downloaded as a NumPy `.npy` array.

The initial five-variable demo is generated locally. Editing a generating graph
does not change any existing observations. Every generated/imported dataset is
immutable. Candidate scores always use that dataset's column mapping.

The entire **Active sample** card opens the selector. Use the trash icon beside
it to **Manage samples**. **Hide** removes a sample from the selector while keeping
its observations and references; **Restore** brings it back. **Delete forever**
permanently removes the sample and all run records that used it, including result
and history files. The confirmation lists affected experiments, which keep their
graphs and settings with no active sample. Exported files remain independent.
Deleting the last sample leaves the editors available with scoring disabled;
the demo sample is not recreated on restart.

Each saved experiment card has a trash button. Permanent experiment deletion
removes its graphs/settings and owned run records/files, while keeping shared
samples available. Deletion requires an explicit confirmation showing the impact;
a changed set of affected records requires reviewing it again. An affected active
run must be stopped before deletion. Local filesystem/database deletions use
recoverable staging so interrupted deletions can be finished or rolled back on
restart. These are permanent removals from the app, not forensic disk erasure.

Select a variable to see a solid outer ring. Other variables in
the same variance group receive dashed rings, the group is highlighted
in the legend, and the inspector lists its peers. Clearing the selection clears
these indicators without changing graph positions or group colors.

## Random models and graph arrangement

Both graph editors have a **Random** menu for a **Random DAG + partition** or a
**Random partition only**. The DAG controls expose `p`, `density` (edge probability
per pair in a uniformly shuffled causal order), `num_colors`, `lambda_min`,
`lambda_max`, `omega_min`, `omega_max`, and `seed`. Defaults match the distributions
in `ColoredNodes/utils.py`: edge probability 0.5, uniform signed coefficient
magnitudes 0.25–1, and uniform group noise variances 0.2–2. As in the original
partition generator, group assignments are independent and some colors may be
unused. The new generators use a local NumPy RNG; seeds repeat within this
implementation, rather than reproducing the legacy global-RNG sequence.

In Explore & score, `p` is fixed to the active sample and variable IDs/names are
preserved. A partition-only change keeps all edges, coefficients, and positions.
Noise bounds specify **variances**, not standard deviations. These actions edit
the model and support Undo; create a sample separately in Create data.

New models use circular positions. **Arrange graph** places existing nodes in a
circle and fits the view. This avoids edges lying along the same straight path
through intermediate nodes; dense graphs can still have edge crossings. Saved
manual layouts remain available. Dragging maintains the canvas state locally and
commits one undoable position change on release.

## Score and statistical conventions

The live counter matches `ColoredNodes.utils.score_DAG`, including its
sample-size normalization and omitted constants. **Higher is better** for
models evaluated on the **same data and centering convention**.

For residual variances `r_i`, nonempty variance blocks `B`, `e` edges, and `n`
observations, the displayed score is:

```text
-.5 * sum_B |B| * (log(mean_{i in B} r_i) + 1)
- log(n)/(2*n) * (e + number_of_blocks)
```

Centering is enabled by default; columns are not standardized. With centering
disabled, regression uses a fixed zero mean. The displayed parameter count is
the covariance-model count (edges plus variance groups). Gaussian constants and
the constant mean-parameter penalty are omitted, consistently with the research
code. A node's parent regressions are cached, while likelihood contributions
are pooled by color block. A changed edge may affect the whole block; optimized
partitions are recomputed globally.

Simulation follows `X_j = sum_i beta_ij X_i + epsilon_j`, with mutually
independent zero-mean Gaussian errors and a shared error variance for nodes in
the same group. This version does not generate discrete/nonlinear models or
handle interventions, missing values, or latent variables.

## Algorithm behavior

- **GES · Gaussian BIC:** standard greedy equivalence search using `ges==1.1.1`.
  Forward and backward phases are enabled by default; optional turning and
  repetition until convergence are available. Uses one variance per variable,
  ignores candidate colors, and converts an initial DAG to its CPDAG before
  searching. Both centered and fixed-zero-mean scoring are supported. Returns
  the CPDAG and a consistent representative DAG with fitted parameters, ready
  for export to Explore & score or the next algorithm. Native BIC is divided by
  sample size and verified against the live scorer. Progress counts accepted
  moves; its activity bar does not estimate a completion percentage. Collinear
  sample columns are rejected.
- **MCMC:** iterations, initial graph/partition, color-vs-edge move probability,
  BIC or most-visited-model selection, optional target BIC stopping, batch size,
  score history, and final partition refinement.
- **Greedy graph + colors:** random restarts, allowed edge/color moves, optional
  initial graph/partition, fixed partition, and final partition refinement.
- **Greedy optimal partition:** random restarts, allowed edge moves, and an
  optional initial DAG. It optimizes the partition at each step; it cannot hold
  a partition fixed.

Fixing a partition disables both color moves and final refinement, even when the
refinement preference was previously on. MCMC's existing `map` mode returns the
most visited state, not a BIC value; the app reports its visit count separately
and independently scores its representative DAG. No burn-in or thinning
controls are advertised because the original implementation does not provide
them. The label follows the existing implementation and does not assert that
visit counts are calibrated posterior probabilities.

The algorithms' partially directed output is displayed as such. The application
also retains the actual representative DAG rather than treating undirected
edges as two directed edges for scoring. Fitted coefficients and block variances
are stored on result graphs. A BIC trace records search states; final partition
refinement can improve the returned score beyond the sampled states.

The original three-value return tuples remain the default. The modified main
functions accept optional `return_dag` and `on_progress` arguments; greedy
methods also accept an initial DAG, and greedy color accepts an initial
partition. Original script-style imports remain supported. Plotting imports in
the greedy-DP demonstration are deferred until that demonstration is run.

## Storage, reproducibility, and limits

Everything is stored under `.daglab/` by default:

```text
.daglab/
  experiments.sqlite3    # dataset metadata, named workspaces, run records
  datasets/<id>.npy       # immutable numeric samples
  runs/<id>/             # frozen request, worker log, progress, result/history
```

The current browser draft is also restored from local storage; use **Save
experiment** for a named durable copy. Datasets and local runtime files are
ignored by Git. They are not uploaded by pushing the repository.

Only one algorithm worker runs at a time. Stopping a run preserves its input
record and last sampled progress but does not produce a final estimate. Stopped
or interrupted runs cannot resume mid-chain. The first use of an algorithm may
spend time compiling its numerical kernels. Seed both Python and NumPy through
the settings; changing algorithm code, dependency versions, or MCMC batch size
can change a trajectory. Runs made with uncommitted code are marked as modified.

Limits in this version: 1–100 variables, 2–1,000,000 observations, 50 MB CSV or
experiment imports, 1–10,000 greedy random restarts, 1–100 million MCMC iterations,
and at most 2 million
iterations when exact per-iteration history is enabled. Practical interactive
limits depend on graph density and hardware. Algorithm runs require more
observations than variables; singular/near-zero-variance models report errors.
The live chart is a bounded sampled trace, not the complete chain. Very large
experiment exports may exceed the import limit; use CSV and saved local data
for those cases.

The server binds to loopback and rejects external origins for writes. It has no
multi-user authentication or remote job queue and is intended for local use.

## Development and extensions

```text
ColoredNodes/          # existing research code, with optional integration hooks
daglab/models.py        # graph, workspace, and settings contracts
daglab/scoring.py       # live scorer and sample generation
daglab/generation.py    # seeded random DAGs, partitions, circular layout
daglab/ges_adapter.py   # Gaussian GES and consistent CPDAG/DAG handling
daglab/algorithms.py    # adapter registry and UI field definitions
daglab/worker.py        # isolated computation process
daglab/jobs.py          # worker lifecycle and immutable run records
daglab/storage.py       # SQLite/files and CSV import
daglab/api.py           # local FastAPI application
frontend/src/          # React + React Flow interface
tests/                 # numerical agreement, integration, persistence, workers
```

To add an algorithm in the current model family, add a registry entry with its
field descriptions and an adapter in `daglab/algorithms.py`, extending
`RunSettings` for any new inputs. The UI renders the declared fields. Adapters
must return the representative DAG, partition, native display structure, and
diagnostics, and verify the reported score against the common scorer.

Other model families (including `ColoredEdges`) should add a family-specific
graph/annotation contract, scorer, and editor controls rather than reusing
node variance groups for a different statistical meaning. Algorithm adapters
are separate from storage, scoring, and rendering to support that expansion.

For UI development, run the Python server on port 8765 and `npm run dev` in
`frontend/`; Vite serves port 5173 and proxies `/api` to Python. For validation:

```sh
.venv/bin/python -m pytest -q
cd frontend && npm run build
npm run test:geometry
```

The Gibbs implementation and other repository sections are not integrated in
this version. They need their own checks for preprocessing, score consistency,
and output semantics before being registered.
