import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  ArrowUpRight,
  Check,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Copy,
  Database,
  FileJson,
  FlaskConical,
  FolderOpen,
  GitBranch,
  Layers3,
  Loader2,
  Play,
  Plus,
  Save,
  Settings2,
  ShieldCheck,
  Square,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import { GraphEditor } from "./GraphEditor";
import { ScoreChart } from "./ScoreChart";
import { SampleManager } from "./SampleManager";
import type {
  Algorithm,
  Dataset,
  Experiment,
  Field,
  Graph,
  Run,
  Score,
  Settings,
  Workspace,
} from "./types";
import {
  api,
  clone,
  download,
  emptyGraph,
  formatScore,
  groupColor,
} from "./types";

type Page = "explore" | "data" | "runs" | "experiments";
type Bootstrap = {
  datasets: Dataset[];
  workspace: Workspace;
  algorithms: Algorithm[];
  experiments: Experiment[];
  runs: Run[];
};
const pageInfo = {
  data: [
    "Create your data",
    "Define a generating model, or bring your own observations.",
  ],
  explore: [
    "Explore a structure",
    "Follow your intuition. Let the score respond.",
  ],
  runs: [
    "Discover the structure",
    "Configure your search and see what the data suggests.",
  ],
  experiments: [
    "Your experiments",
    "Pick up where you left off. Every model has a story.",
  ],
};
const elapsed = (seconds: number) =>
  seconds < 60
    ? `${seconds.toFixed(1)}s`
    : `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
function dateLabel(value: string) {
  return new Date(value).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function SettingField({
  field,
  settings,
  onChange,
}: {
  field: Field;
  settings: Settings;
  onChange: (patch: Partial<Settings>) => void;
}) {
  if (
    field.visible_if &&
    Object.entries(field.visible_if).some(
      ([key, value]) => settings[key as keyof Settings] !== value,
    )
  )
    return null;
  const value =
    settings[field.name] ??
    (field.name === "ges_phases"
      ? ["forward", "backward"]
      : field.name === "ges_iterate"
        ? false
        : undefined);
  if (field.type === "boolean")
    return (
      <label className="toggle-field">
        <span>
          <strong>{field.label}</strong>
          {field.hint && <small>{field.hint}</small>}
        </span>
        <input
          type="checkbox"
          checked={Boolean(value)}
          onChange={(e) => onChange({ [field.name]: e.target.checked })}
        />
      </label>
    );
  if (field.type === "multiselect")
    return (
      <fieldset className="move-field">
        <legend>{field.label}</legend>
        <div className="move-grid">
          {field.options?.map((option) => {
            const disabled =
              settings.fixed_partition && option.value === "change_color";
            return (
              <label key={option.value} className={disabled ? "disabled" : ""}>
                <input
                  type="checkbox"
                  checked={
                    !disabled && (value as string[]).includes(option.value)
                  }
                  disabled={disabled}
                  onChange={(e) =>
                    onChange({
                      [field.name]: e.target.checked
                        ? field
                            .options!.filter(
                              (o) =>
                                (value as string[]).includes(o.value) ||
                                o.value === option.value,
                            )
                            .map((o) => o.value)
                        : (value as string[]).filter((v) => v !== option.value),
                    })
                  }
                />
                {option.label}
              </label>
            );
          })}
        </div>
      </fieldset>
    );
  return (
    <label>
      {field.label}
      {field.type === "select" ? (
        <select
          value={String(value)}
          onChange={(e) =>
            onChange(
              field.name === "mode"
                ? { mode: e.target.value as Settings["mode"], target_bic: null }
                : { [field.name]: e.target.value },
            )
          }
        >
          {field.options?.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      ) : (
        <input
          type="number"
          min={field.min}
          max={field.max}
          step={field.step ?? "any"}
          value={value === null ? "" : String(value)}
          placeholder={
            field.type === "optional_number" ? "No target" : undefined
          }
          onChange={(e) => {
            if (e.target.value === "" && field.type === "optional_number")
              onChange({ [field.name]: null });
            else if (
              e.target.value !== "" &&
              Number.isFinite(Number(e.target.value))
            )
              onChange({ [field.name]: Number(e.target.value) });
          }}
        />
      )}
      {field.hint && <small>{field.hint}</small>}
    </label>
  );
}

export default function App() {
  const [workspace, setWorkspace] = useState<Workspace | null>(null),
    [datasets, setDatasets] = useState<Dataset[]>([]),
    [algorithms, setAlgorithms] = useState<Algorithm[]>([]);
  const [experiments, setExperiments] = useState<Experiment[]>([]),
    [runs, setRuns] = useState<Run[]>([]),
    [page, setPage] = useState<Page>("explore");
  const [error, setError] = useState(""),
    [toast, setToast] = useState<{ message: string; error: boolean } | null>(
      null,
    ),
    [busy, setBusy] = useState("");
  const [savedId, setSavedId] = useState<string | null>(null),
    [savedSnapshot, setSavedSnapshot] = useState("");
  const [score, setScore] = useState<Score | null>(null),
    [scoreError, setScoreError] = useState(""),
    [scoring, setScoring] = useState(false),
    [truthScore, setTruthScore] = useState<number | undefined>();
  const [selectedRun, setSelectedRun] = useState<string | null>(null),
    [resultView, setResultView] = useState<"dag" | "cpdag">("dag");
  const [sampleSize, setSampleSize] = useState(1000),
    [sampleSeed, setSampleSeed] = useState(42),
    [sampleName, setSampleName] = useState("Generated sample"),
    [csvHeader, setCsvHeader] = useState(true);
  const [help, setHelp] = useState(false);
  const [sampleManager, setSampleManager] = useState(false);
  const [allSamples, setAllSamples] = useState<Dataset[]>([]);
  const csvInput = useRef<HTMLInputElement>(null),
    experimentInput = useRef<HTMLInputElement>(null);
  const defaultWorkspace = useRef<Workspace | null>(null);
  const ready = useRef(false);
  const latestWorkspace = useRef<Workspace | null>(null);
  latestWorkspace.current = workspace;
  const notify = useCallback(
    (message: string, isError = false) => setToast({ message, error: isError }),
    [],
  );
  const fail = useCallback(
    (message: string) => notify(message, true),
    [notify],
  );
  const update = useCallback(
    (patch: Partial<Workspace>) =>
      setWorkspace((w) => (w ? { ...w, ...patch } : w)),
    [],
  );
  const updateSettings = useCallback(
    (patch: Partial<Settings>) =>
      setWorkspace((w) =>
        w ? { ...w, settings: { ...w.settings, ...patch } } : w,
      ),
    [],
  );
  const setCandidate = useCallback(
    (graph: Graph) => update({ candidate_graph: graph }),
    [update],
  );
  const setSource = useCallback(
    (graph: Graph) => update({ source_graph: graph }),
    [update],
  );
  const dataset = datasets.find((d) => d.id === workspace?.dataset_id);
  const algorithm = algorithms.find((a) => a.id === workspace?.algorithm);
  const currentRun = runs.find((r) => r.id === selectedRun) || runs[0];
  const activeRun = runs.find((r) => r.status === "running");
  const dirty = workspace ? JSON.stringify(workspace) !== savedSnapshot : false;

  useEffect(() => {
    let cancelled = false;
    api<Bootstrap>("/bootstrap")
      .then((data) => {
        if (cancelled) return;
        defaultWorkspace.current = clone(data.workspace);
        let initial = data.workspace;
        try {
          const draft = JSON.parse(
            localStorage.getItem("daglab-draft-v1") || "null",
          );
          if (
            draft?.schema_version === 1 &&
            draft.candidate_graph?.nodes?.length &&
            data.datasets.some((d) => d.id === draft.dataset_id)
          )
            initial = draft;
        } catch {
          /* The server workspace is the fallback for an invalid local draft. */
        }
        setWorkspace(initial);
        setDatasets(data.datasets);
        setAlgorithms(data.algorithms);
        setExperiments(data.experiments);
        setRuns(data.runs);
        ready.current = true;
      })
      .catch((err) => !cancelled && setError(err.message));
    return () => {
      cancelled = true;
    };
  }, []);
  useEffect(() => {
    if (workspace && ready.current) {
      try {
        localStorage.setItem("daglab-draft-v1", JSON.stringify(workspace));
      } catch {
        /* Explicit save remains available when browser storage is full. */
      }
    }
  }, [workspace]);
  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(null), 7000);
    return () => clearTimeout(timer);
  }, [toast]);
  const scoreKey = useMemo(
    () =>
      workspace
        ? JSON.stringify({
            dataset: workspace.dataset_id,
            center: workspace.center,
            nodes: workspace.candidate_graph.nodes.map(
              ({ id, label, group }) => ({ id, label, group }),
            ),
            edges: workspace.candidate_graph.edges.map(
              ({ source, target }) => ({ source, target }),
            ),
          })
        : "",
    [workspace],
  );
  useEffect(() => {
    if (!workspace?.dataset_id) {
      setScore(null);
      setScoreError("");
      setScoring(false);
      return;
    }
    const controller = new AbortController();
    setScoring(true);
    setScoreError("");
    const timer = setTimeout(
      () =>
        api<Score>("/score", {
          method: "POST",
          signal: controller.signal,
          body: JSON.stringify({
            dataset_id: workspace.dataset_id,
            graph: workspace.candidate_graph,
            center: workspace.center,
          }),
        })
          .then((value) => {
            if (!controller.signal.aborted) {
              setScore(value);
              setScoring(false);
            }
          })
          .catch((err) => {
            if (!controller.signal.aborted) {
              setScore(null);
              setScoreError(err.message);
              setScoring(false);
            }
          }),
      180,
    );
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
    // Positions and generating coefficients do not affect a candidate's BIC.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scoreKey]);
  useEffect(() => {
    setTruthScore(undefined);
    if (!dataset?.provenance.graph) return;
    const controller = new AbortController();
    api<Score>("/score", {
      method: "POST",
      signal: controller.signal,
      body: JSON.stringify({
        dataset_id: dataset.id,
        graph: dataset.provenance.graph,
        center: workspace?.center,
      }),
    })
      .then((value) => setTruthScore(value.score))
      .catch(() => {});
    return () => controller.abort();
  }, [dataset, workspace?.center]);
  useEffect(() => {
    if (!activeRun) return;
    let cancelled = false,
      inFlight = false;
    const timer = setInterval(() => {
      if (inFlight) return;
      inFlight = true;
      api<Run>(`/runs/${activeRun.id}`)
        .then((value) => {
          if (!cancelled)
            setRuns((rs) =>
              rs.map((run) => (run.id === value.id ? value : run)),
            );
        })
        .catch((err) => {
          if (!cancelled) fail(err.message);
        })
        .finally(() => {
          inFlight = false;
        });
    }, 900);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [activeRun?.id, fail]);

  async function action(name: string, operation: () => Promise<void>) {
    setBusy(name);
    try {
      await operation();
    } catch (err) {
      fail((err as Error).message);
    } finally {
      setBusy("");
    }
  }
  function activateDataset(next: Dataset) {
    if (!workspace) return;
    const graph = next.provenance.graph
      ? { ...clone(next.provenance.graph), edges: [] }
      : emptyGraph(next);
    update({ dataset_id: next.id, candidate_graph: graph });
  }
  async function manageSamples() {
    await action("samples", async () => {
      setAllSamples(await api<Dataset[]>("/datasets?include_deleted=true"));
      setSampleManager(true);
    });
  }
  async function deleteSample(sample: Dataset) {
    await action("deleteSample", async () => {
      const deleted = await api<Dataset>(`/datasets/${sample.id}/delete`, {
        method: "POST",
      });
      const remaining = datasets.filter((item) => item.id !== sample.id);
      setDatasets(remaining);
      setAllSamples((items) =>
        items.map((item) => (item.id === sample.id ? deleted : item)),
      );
      if (latestWorkspace.current?.dataset_id === sample.id) {
        if (remaining.length) activateDataset(remaining[0]);
        else update({ dataset_id: null });
      }
      notify(`Deleted ${sample.name}. It can be restored in Manage samples.`);
    });
  }
  async function restoreSample(sample: Dataset) {
    await action("restoreSample", async () => {
      const restored = await api<Dataset>(`/datasets/${sample.id}/restore`, {
        method: "POST",
      });
      setDatasets((items) => [
        restored,
        ...items.filter((item) => item.id !== restored.id),
      ]);
      setAllSamples((items) =>
        items.map((item) => (item.id === sample.id ? restored : item)),
      );
      if (!latestWorkspace.current?.dataset_id) activateDataset(restored);
      notify(`Restored ${sample.name}.`);
    });
  }
  async function ensureSample(id: string) {
    let sample =
      datasets.find((item) => item.id === id) ||
      (await api<Dataset>(`/datasets/${id}`));
    if (sample.deleted_at)
      sample = await api<Dataset>(`/datasets/${id}/restore`, {
        method: "POST",
      });
    setDatasets((items) => [
      sample,
      ...items.filter((item) => item.id !== sample.id),
    ]);
    setAllSamples((items) =>
      items.map((item) => (item.id === sample.id ? sample : item)),
    );
    return sample;
  }
  async function generate() {
    if (!workspace) return;
    await action("generate", async () => {
      const next = await api<Dataset>("/datasets/generate", {
        method: "POST",
        body: JSON.stringify({
          graph: workspace.source_graph,
          sample_size: sampleSize,
          seed: sampleSeed,
          name: sampleName,
        }),
      });
      setDatasets((ds) => [next, ...ds]);
      activateDataset(next);
      notify(
        `Generated ${next.n.toLocaleString()} observations. Your candidate is ready to explore.`,
      );
    });
  }
  async function importCsv(file: File) {
    await action("import", async () => {
      const form = new FormData();
      form.append("file", file);
      const next = await api<Dataset>(`/datasets/import?header=${csvHeader}`, {
        method: "POST",
        body: form,
      });
      setDatasets((ds) => [next, ...ds]);
      activateDataset(next);
      notify(`Imported ${next.name}.`);
    });
  }
  async function save() {
    if (!workspace) return;
    await action("save", async () => {
      const record = await api<Experiment>(
        savedId ? `/workspaces/${savedId}` : "/workspaces",
        { method: savedId ? "PUT" : "POST", body: JSON.stringify(workspace) },
      );
      setSavedId(record.id);
      setSavedSnapshot(JSON.stringify(workspace));
      setExperiments((es) => [record, ...es.filter((e) => e.id !== record.id)]);
      notify("Experiment saved on this computer.");
    });
  }
  async function exportWorkspace() {
    if (!workspace) return;
    await action("export", async () => {
      const bundle = await api("/workspaces/export", {
        method: "POST",
        body: JSON.stringify(workspace),
      });
      download(
        bundle,
        `${workspace.name.replace(/[^a-zA-Z0-9_-]/g, "-") || "experiment"}.json`,
      );
      notify("Exported the experiment with its sample and model settings.");
    });
  }
  async function importExperiment(file: File) {
    await action("importExperiment", async () => {
      const form = new FormData();
      form.append("file", file);
      const record = await api<Experiment>("/workspaces/import", {
        method: "POST",
        body: form,
      });
      setExperiments((es) => [record, ...es]);
      setDatasets(await api<Dataset[]>("/datasets"));
      await applyExperiment(record);
      notify("Experiment imported.");
    });
  }
  async function applyExperiment(record: Experiment) {
    if (record.workspace.dataset_id)
      await ensureSample(record.workspace.dataset_id);
    setWorkspace(clone(record.workspace));
    setSavedId(record.id);
    setSavedSnapshot(JSON.stringify(record.workspace));
    setPage("explore");
  }
  function openExperiment(record: Experiment) {
    void action("open", () => applyExperiment(record));
  }
  async function optimizePartition() {
    if (!workspace?.dataset_id) return;
    await action("partition", async () => {
      const optimized = await api<Score>("/score", {
        method: "POST",
        body: JSON.stringify({
          dataset_id: workspace.dataset_id,
          graph: workspace.candidate_graph,
          center: workspace.center,
          optimize_partition: true,
        }),
      });
      const latest = latestWorkspace.current;
      if (
        !latest ||
        latest.dataset_id !== workspace.dataset_id ||
        latest.center !== workspace.center ||
        JSON.stringify(
          latest.candidate_graph.nodes.map(({ id, label, group }) => ({
            id,
            label,
            group,
          })),
        ) !==
          JSON.stringify(
            workspace.candidate_graph.nodes.map(({ id, label, group }) => ({
              id,
              label,
              group,
            })),
          ) ||
        JSON.stringify(
          latest.candidate_graph.edges.map(({ source, target }) => ({
            source,
            target,
          })),
        ) !==
          JSON.stringify(
            workspace.candidate_graph.edges.map(({ source, target }) => ({
              source,
              target,
            })),
          )
      )
        throw new Error(
          "The candidate changed while optimizing. Please try again.",
        );
      const graph = clone(latest.candidate_graph);
      optimized.blocks.forEach((block, i) => {
        for (const node of graph.nodes)
          if (block.nodes.includes(node.id)) node.group = String(i + 1);
        graph.variances[String(i + 1)] = block.variance;
      });
      setCandidate(graph);
      notify(`Optimized partition: ${optimized.groups} variance groups.`);
    });
  }
  async function startRun() {
    if (!workspace) return;
    await action("run", async () => {
      const run = await api<Run>("/runs", {
        method: "POST",
        body: JSON.stringify({
          dataset_id: workspace.dataset_id,
          graph: workspace.candidate_graph,
          algorithm: workspace.algorithm,
          settings: { ...workspace.settings, center: workspace.center },
        }),
      });
      setRuns((rs) => [run, ...rs]);
      setSelectedRun(run.id);
      setResultView("dag");
      notify("Search started in a separate process. You can keep editing.");
    });
  }
  async function adoptRun(run: Run, stayInRuns = false) {
    if (!run.result) return;
    await action("adopt", async () => {
      await ensureSample(run.request.dataset_id);
      setWorkspace((current) =>
        current
          ? {
              ...current,
              dataset_id: run.request.dataset_id,
              candidate_graph: clone(run.result!.graph),
              center: run.request.settings.center,
              settings: {
                ...current.settings,
                center: run.request.settings.center,
                start_from_candidate: true,
              },
            }
          : current,
      );
      if (!stayInRuns) setPage("explore");
      notify(
        stayInRuns
          ? "Result set as the DAG in Explore & score and the starting point for your next run. Choose an algorithm, then run it."
          : "Result sent to Explore & score. Your next run can start from this DAG.",
      );
    });
  }

  if (!workspace)
    return (
      <div className="loading-screen">
        <div className="brand-symbol">
          <GitBranch />
        </div>
        <h2>DAG Lab</h2>
        {error ? (
          <>
            <p>{error}</p>
            <button className="primary" onClick={() => location.reload()}>
              Try again
            </button>
          </>
        ) : (
          <>
            <Loader2 className="spin" size={20} />
            <p>Preparing your workbench…</p>
          </>
        )}
      </div>
    );
  const groupNames = [
    ...new Set(workspace.candidate_graph.nodes.map((n) => n.group)),
  ];
  const result = currentRun?.result;
  const resultTrace = result?.trace || currentRun?.progress?.trace || [];
  const compatibleTruth = dataset?.provenance.graph;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            setPage("explore");
          }}
        >
          <span className="brand-symbol">
            <GitBranch size={22} />
          </span>
          <span>
            DAG<span className="brand-light"> Lab</span>
            <small>STRUCTURE DISCOVERY</small>
          </span>
        </a>
        <div className="sidebar-section-label">WORKBENCH</div>
        <nav>
          {(
            [
              { id: "data", icon: Database, label: "Create data", step: "01" },
              {
                id: "explore",
                icon: GitBranch,
                label: "Explore & score",
                step: "02",
              },
              { id: "runs", icon: Play, label: "Run algorithms", step: "03" },
            ] as const
          ).map((item) => (
            <button
              key={item.id}
              className={page === item.id ? "active" : ""}
              onClick={() => setPage(item.id)}
            >
              <item.icon size={18} />
              <span>{item.label}</span>
              <small>{item.step}</small>
              {item.id === "runs" && activeRun && <i className="running-dot" />}
            </button>
          ))}
        </nav>
        <div className="sidebar-rule" />
        <nav>
          <button
            className={page === "experiments" ? "active" : ""}
            onClick={() => setPage("experiments")}
          >
            <FolderOpen size={18} />
            <span>Experiments</span>
            <small>{experiments.length}</small>
          </button>
        </nav>
        <div className="sidebar-bottom">
          <div className="local-badge">
            <span className="status-dot" />
            <span>
              Running locally<small>Your data stays on this computer</small>
            </span>
          </div>
          <button className="sidebar-help" onClick={() => setHelp(true)}>
            <CircleHelp size={16} />
            <span>About the workbench</span>
            <ArrowUpRight size={14} />
          </button>
          <div className="sidebar-version">
            DAGdiscovery<span>v0.1</span>
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            <FlaskConical size={15} />
            <span>Experiments</span>
            <ChevronRight size={13} />
            <input
              aria-label="Experiment name"
              value={workspace.name}
              maxLength={120}
              onChange={(e) => update({ name: e.target.value })}
            />
            <span
              className={`save-state ${dirty ? "dirty" : ""}`}
              title="Save creates a named experiment; your current draft is also restored in this browser."
            >
              {dirty ? "Draft" : "Saved"}
            </span>
          </div>
          <div className="topbar-actions">
            <button
              className="secondary"
              onClick={exportWorkspace}
              disabled={!!busy}
            >
              <ArrowDownToLine size={15} />
              <span>Export</span>
            </button>
            <button
              className="primary"
              onClick={save}
              disabled={!!busy || !workspace.name.trim()}
            >
              {busy === "save" ? (
                <Loader2 className="spin" size={15} />
              ) : (
                <Save size={15} />
              )}
              <span>Save experiment</span>
            </button>
          </div>
        </header>
        <main>
          <div className="page-heading">
            <div>
              <div className="eyebrow">YOUR CAUSAL WORKBENCH</div>
              <h1>
                {pageInfo[page][0]}
                <span className="heading-dot">.</span>
              </h1>
              <p>{pageInfo[page][1]}</p>
            </div>
            {page !== "experiments" && (
              <div className="sample-controls">
                <div className="dataset-selector">
                  <Database size={16} />
                  <div>
                    <small>ACTIVE SAMPLE</small>
                    <span className="dataset-name">
                      {dataset?.name || "No active sample"}
                    </span>
                    <select
                      aria-label="Active sample"
                      title="Choose the active sample"
                      value={workspace.dataset_id || ""}
                      disabled={!datasets.length}
                      onChange={(e) => {
                        const next = datasets.find(
                          (d) => d.id === e.target.value,
                        );
                        if (next) activateDataset(next);
                      }}
                    >
                      {!datasets.length && (
                        <option value="">No active sample</option>
                      )}
                      {datasets.map((d) => (
                        <option key={d.id} value={d.id}>
                          {d.name}
                        </option>
                      ))}
                    </select>
                  </div>
                  <ChevronDown size={14} />
                </div>
                <button
                  className="secondary manage-samples"
                  aria-label="Manage samples"
                  title="Manage samples"
                  onClick={manageSamples}
                  disabled={!!busy}
                >
                  <Trash2 size={16} />
                </button>
              </div>
            )}
          </div>

          {page === "explore" && (
            <>
              {!dataset && (
                <div className="info-box no-sample-message">
                  Generate, import, or restore a sample to see live BIC scores.
                </div>
              )}
              <div className="score-row">
                <section
                  className={`score-card main-score ${scoring ? "updating" : ""}`}
                >
                  <div className="score-label">
                    BIC SCORE{" "}
                    <span className="live-badge">
                      <span />
                      {!dataset ? "No sample" : scoring ? "Updating" : "Live"}
                    </span>
                  </div>
                  <div className="score-value">
                    {formatScore(score?.score)}
                    <ArrowUpRight size={22} />
                  </div>
                  <div className="score-caption">
                    Higher is better <span>·</span> normalized Gaussian BIC
                  </div>
                </section>
                <section className="score-card">
                  <div className="score-label">
                    MODEL COMPLEXITY
                    <Layers3 size={15} />
                  </div>
                  <div className="complexity-value">
                    <strong>{workspace.candidate_graph.edges.length}</strong>
                    <span>edges</span>
                    <i />
                    <strong>{groupNames.length}</strong>
                    <span>groups</span>
                  </div>
                  <div className="score-caption">
                    {score?.parameters ??
                      workspace.candidate_graph.edges.length +
                        groupNames.length}{" "}
                    free covariance parameters
                  </div>
                </section>
                <section className="score-card">
                  <div className="score-label">
                    REFERENCE MODEL
                    <GitBranch size={15} />
                  </div>
                  <div className="reference-value">
                    {truthScore === undefined
                      ? "No known DAG"
                      : formatScore(truthScore)}
                  </div>
                  <div className="score-caption">
                    {truthScore === undefined
                      ? "Available for generated samples"
                      : `Generating DAG · candidate Δ ${score ? (score.score - truthScore >= 0 ? "+" : "") + (score.score - truthScore).toFixed(5) : "—"}`}
                  </div>
                </section>
              </div>
              {scoreError && (
                <div className="error-banner">
                  <CircleHelp size={16} />
                  <span>{scoreError}</span>
                </div>
              )}
              <div className="section-heading">
                <div>
                  <h2>Candidate model</h2>
                  <span className="muted small">
                    Your structure, evaluated against the active sample
                  </span>
                </div>
                <div className="section-actions">
                  <label className="inline-check">
                    <input
                      type="checkbox"
                      checked={workspace.center}
                      onChange={(e) => {
                        update({ center: e.target.checked });
                        updateSettings({ center: e.target.checked });
                      }}
                    />
                    Center data
                  </label>
                  <button
                    className="secondary"
                    onClick={() => {
                      if (!compatibleTruth) {
                        fail("This sample has no saved generating model.");
                        return;
                      }
                      setCandidate(clone(compatibleTruth));
                      notify("Copied the sample’s generating model.");
                    }}
                    disabled={!compatibleTruth}
                  >
                    <Copy size={14} />
                    Copy reference
                  </button>
                </div>
              </div>
              <GraphEditor
                key={`candidate-${workspace.dataset_id}-${savedId}`}
                graph={workspace.candidate_graph}
                onChange={setCandidate}
                onError={fail}
                footer={
                  <>
                    <div className="eyebrow">PARTITION EXPLORER</div>
                    <p className="small muted">
                      Keep the edges and find the variance grouping with the
                      best BIC.
                    </p>
                    <button
                      className="secondary full"
                      onClick={optimizePartition}
                      disabled={!!busy || !dataset}
                    >
                      {busy === "partition" ? (
                        <Loader2 size={14} className="spin" />
                      ) : (
                        <Layers3 size={14} />
                      )}
                      Optimize partition
                    </button>
                  </>
                }
              />
              <div className="bottom-note">
                <ShieldCheck size={14} />
                <span>
                  Only acyclic structures are allowed. Scores update when edges
                  or colors change.
                </span>
                <button onClick={() => setPage("runs")}>
                  Try an algorithm
                  <ArrowRight size={14} />
                </button>
              </div>
            </>
          )}

          {page === "data" && (
            <>
              <div className="data-intro">
                <div className="intro-icon">
                  <Database size={20} />
                </div>
                <div>
                  <strong>
                    Start with a model. Or start with observations.
                  </strong>
                  <p>
                    Draw a linear Gaussian DAG below, then sample from it. CSV
                    imports can be used directly.
                  </p>
                </div>
                <div className="import-action">
                  <button
                    className="secondary"
                    onClick={() => csvInput.current?.click()}
                    disabled={!!busy}
                  >
                    <Upload size={15} />
                    Import CSV
                  </button>
                  <label className="inline-check">
                    <input
                      type="checkbox"
                      checked={csvHeader}
                      onChange={(e) => setCsvHeader(e.target.checked)}
                    />
                    First row is a header
                  </label>
                </div>
              </div>
              <GraphEditor
                key={`source-${savedId}`}
                graph={workspace.source_graph}
                source
                onChange={setSource}
                onError={fail}
                title="Generating DAG"
                footer={
                  <>
                    <div className="eyebrow">GENERATE OBSERVATIONS</div>
                    <label>
                      Sample name
                      <input
                        value={sampleName}
                        maxLength={120}
                        onChange={(e) => setSampleName(e.target.value)}
                      />
                    </label>
                    <div className="field-pair">
                      <label>
                        Observations
                        <input
                          type="number"
                          min="2"
                          max="1000000"
                          step="100"
                          value={sampleSize}
                          onChange={(e) =>
                            setSampleSize(Number(e.target.value))
                          }
                        />
                      </label>
                      <label>
                        Random seed
                        <input
                          type="number"
                          min="0"
                          max={2 ** 32 - 1}
                          value={sampleSeed}
                          onChange={(e) =>
                            setSampleSeed(Number(e.target.value))
                          }
                        />
                      </label>
                    </div>
                    <button
                      className="primary full"
                      onClick={generate}
                      disabled={!!busy || sampleSize < 2 || !sampleName.trim()}
                    >
                      {busy === "generate" ? (
                        <Loader2 size={15} className="spin" />
                      ) : (
                        <FlaskConical size={15} />
                      )}
                      Generate sample
                    </button>
                    <small className="sampling-note">
                      This saves a new sample. Existing observations stay as
                      they are.
                    </small>
                  </>
                }
              />
              {dataset && (
                <section className="data-preview">
                  <div className="section-heading">
                    <div>
                      <h2>{dataset.name}</h2>
                      <span className="muted small">
                        {dataset.n.toLocaleString()} observations · {dataset.p}{" "}
                        variables · {dataset.provenance.type}
                        {dataset.provenance.seed !== undefined
                          ? ` · seed ${dataset.provenance.seed}`
                          : ""}
                      </span>
                    </div>
                    <a
                      className="secondary"
                      href={`/api/datasets/${dataset.id}/csv`}
                      download
                    >
                      <ArrowDownToLine size={14} />
                      Export CSV
                    </a>
                  </div>
                  <div className="table-scroll">
                    <table>
                      <thead>
                        <tr>
                          <th>#</th>
                          {dataset.columns.map((c) => (
                            <th key={c.id}>{c.label}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {dataset.preview.map((row, i) => (
                          <tr key={i}>
                            <td>{i + 1}</td>
                            {row.map((value, j) => (
                              <td key={j}>{value.toFixed(4)}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="table-foot">
                    Previewing the first {dataset.preview.length} observations
                  </div>
                </section>
              )}
            </>
          )}

          {page === "runs" && (
            <div className="runs-layout">
              <aside className="run-config">
                <div className="panel-heading">
                  <span className="eyebrow">SEARCH CONFIGURATION</span>
                  <Settings2 size={15} />
                </div>
                <label>
                  Algorithm
                  <select
                    value={workspace.algorithm}
                    onChange={(e) => {
                      const id = e.target.value;
                      update({ algorithm: id });
                      if (id === "ges")
                        updateSettings({ fixed_partition: false });
                      if (id === "greedy_dp")
                        updateSettings({
                          fixed_partition: false,
                          moves: workspace.settings.moves.filter(
                            (m) => m !== "change_color",
                          ),
                        });
                    }}
                  >
                    {algorithms.map((a) => (
                      <option key={a.id} value={a.id}>
                        {a.name}
                      </option>
                    ))}
                  </select>
                </label>
                <p className="algorithm-description">
                  {algorithm?.description}
                </p>
                <div className="starting-dag" aria-label="Starting DAG summary">
                  <GitBranch size={17} />
                  <div>
                    <strong>
                      {workspace.settings.start_from_candidate
                        ? "Starting DAG: Explore & score"
                        : "Fresh start"}
                    </strong>
                    <span>
                      {workspace.settings.start_from_candidate
                        ? `${workspace.candidate_graph.nodes.length} variables · ${workspace.candidate_graph.edges.length} edges${workspace.algorithm === "ges" ? " · colors ignored; converted to CPDAG" : ` · ${groupNames.length} groups`}`
                        : workspace.algorithm === "ges"
                          ? "Starts from the empty equivalence class."
                          : workspace.algorithm === "mcmc"
                            ? "Empty graph; separate variance groups unless the partition is fixed."
                            : "Random initial graphs for the restarts."}
                    </span>
                  </div>
                  <button
                    aria-label="View starting DAG in Explore and score"
                    title="View DAG in Explore & score"
                    onClick={() => setPage("explore")}
                  >
                    <ArrowUpRight size={16} />
                  </button>
                </div>
                <div className="config-fields">
                  {algorithm?.fields.map((field) => (
                    <SettingField
                      key={field.name}
                      field={field}
                      settings={{
                        ...workspace.settings,
                        center: workspace.center,
                      }}
                      onChange={(patch) => {
                        updateSettings(patch);
                        if (patch.center !== undefined)
                          update({ center: patch.center });
                      }}
                    />
                  ))}
                </div>
                {workspace.algorithm === "mcmc" &&
                  workspace.settings.mode === "map" && (
                    <div className="info-box">
                      MAP mode selects the most visited model. Its visit count
                      and its BIC are reported separately.
                    </div>
                  )}
                <div className="config-source">{algorithm?.source}</div>
                <button
                  className="primary full run-button"
                  onClick={startRun}
                  disabled={
                    !!busy || !!activeRun || !!scoreError || !dataset || scoring
                  }
                >
                  {busy === "run" ? (
                    <Loader2 size={16} className="spin" />
                  ) : (
                    <Play size={16} />
                  )}
                  Run algorithm
                </button>
                {activeRun && (
                  <button
                    className="secondary full stop-button"
                    onClick={() =>
                      action("cancel", async () => {
                        const cancelled = await api<Run>(
                          `/runs/${activeRun.id}/cancel`,
                          { method: "POST" },
                        );
                        setRuns((rs) =>
                          rs.map((r) =>
                            r.id === cancelled.id ? cancelled : r,
                          ),
                        );
                        notify(
                          "Run stopped. Its settings and sampled trace are saved.",
                        );
                      })
                    }
                  >
                    <Square size={13} />
                    Stop active run
                  </button>
                )}
              </aside>
              <div className="run-results">
                {!currentRun ? (
                  <section className="empty-results">
                    <div className="empty-orbit">
                      <GitBranch size={42} />
                      <i />
                      <i />
                    </div>
                    <span className="eyebrow">FROM DATA TO STRUCTURE</span>
                    <h2>Your next discovery starts here</h2>
                    <p>
                      Choose an algorithm and its settings. The search runs
                      separately, so you can continue exploring while it works.
                    </p>
                    <div className="empty-steps">
                      <span>
                        <Check size={14} />
                        Sample ready
                      </span>
                      <span>
                        <Check size={14} />
                        Candidate ready
                      </span>
                      <span>
                        <Play size={13} />
                        Ready to run
                      </span>
                    </div>
                  </section>
                ) : (
                  <>
                    <section className="run-summary">
                      <div className="section-heading">
                        <div>
                          <div className="eyebrow">{currentRun.name}</div>
                          <h2>
                            {currentRun.status === "completed"
                              ? "Search complete"
                              : currentRun.status === "running"
                                ? "Finding a structure"
                                : currentRun.status === "failed"
                                  ? "Search needs attention"
                                  : currentRun.status === "cancelled"
                                    ? "Search stopped"
                                    : "Run interrupted"}
                          </h2>
                        </div>
                        <div className="run-heading-actions">
                          {result && (
                            <button
                              className="primary"
                              disabled={!!busy}
                              onClick={() => void adoptRun(currentRun)}
                            >
                              <ArrowUpRight size={14} />
                              Send to Explore & score
                            </button>
                          )}
                          <span className={`run-status ${currentRun.status}`}>
                            {currentRun.status === "running" ? (
                              <Loader2 size={13} className="spin" />
                            ) : currentRun.status === "completed" ? (
                              <Check size={13} />
                            ) : (
                              <Square size={11} />
                            )}{" "}
                            {currentRun.status}
                          </span>
                        </div>
                      </div>
                      <p className="small muted">
                        Sample:{" "}
                        {datasets.find(
                          (d) => d.id === currentRun.request.dataset_id,
                        )?.name || currentRun.request.dataset_id}{" "}
                        ·{" "}
                        {currentRun.request.settings.center
                          ? "Centered"
                          : "Fixed zero mean"}
                      </p>
                      <div className="run-stat-row">
                        <div>
                          <small>
                            {currentRun.status === "running"
                              ? "BEST BIC"
                              : "RESULT BIC"}
                          </small>
                          <strong>
                            {formatScore(
                              result?.score ?? currentRun.progress?.best_score,
                            )}
                          </strong>
                        </div>
                        <div>
                          <small>
                            {currentRun.request.algorithm === "mcmc"
                              ? "ITERATIONS"
                              : "SEARCH STEPS"}
                          </small>
                          <strong>
                            {(
                              result?.trace.at(-1)?.iteration ??
                              currentRun.progress?.iteration ??
                              0
                            ).toLocaleString()}
                          </strong>
                        </div>
                        <div>
                          <small>ELAPSED</small>
                          <strong>
                            {elapsed(
                              result?.elapsed ??
                                currentRun.progress?.elapsed ??
                                0,
                            )}
                          </strong>
                        </div>
                        {result?.diagnostics.visit_count !== undefined && (
                          <div>
                            <small>MODEL VISITS</small>
                            <strong>
                              {result.diagnostics.visit_count?.toLocaleString()}
                            </strong>
                          </div>
                        )}
                      </div>
                      {currentRun.status === "running" && (
                        <div
                          className={`run-progress ${currentRun.request.algorithm === "ges" ? "indeterminate" : ""}`}
                        >
                          <span
                            style={{
                              width:
                                currentRun.request.algorithm === "ges"
                                  ? "35%"
                                  : currentRun.request.algorithm === "mcmc"
                                    ? `${Math.min(100, (100 * (currentRun.progress?.iteration ?? 0)) / currentRun.request.settings.num_iters)}%`
                                    : `${Math.min(97, (100 * (currentRun.progress?.restart ?? 0)) / currentRun.request.settings.num_starts)}%`,
                            }}
                          />
                          <small>
                            {currentRun.progress?.phase || "Starting worker"}
                            {currentRun.progress?.restart
                              ? ` · restart ${currentRun.progress.restart}/${currentRun.request.settings.num_starts}`
                              : currentRun.request.algorithm === "ges"
                                ? ` · ${currentRun.progress?.iteration ?? 0} accepted moves`
                                : ""}
                          </small>
                        </div>
                      )}
                      {currentRun.error && (
                        <div className="error-banner">{currentRun.error}</div>
                      )}
                      <div className="chart-heading">
                        <span>Score over the search</span>
                        <span>
                          <i className="trace-current" />
                          Current <i className="trace-best" />
                          Best
                        </span>
                      </div>
                      <ScoreChart
                        trace={resultTrace}
                        unit={
                          currentRun.request.algorithm === "ges"
                            ? "accepted moves"
                            : currentRun.request.algorithm === "mcmc"
                              ? "iterations"
                              : "search steps"
                        }
                      />
                      {result && (
                        <div className="result-actions">
                          <button
                            className="primary"
                            onClick={() => void adoptRun(currentRun, true)}
                            disabled={!!busy}
                          >
                            <Copy size={14} />
                            Use as next starting DAG
                          </button>
                          <a
                            className="secondary"
                            href={`/api/runs/${currentRun.id}/export`}
                            download
                          >
                            <ArrowDownToLine size={14} />
                            Run details
                          </a>
                          {result.history_available && (
                            <a
                              className="secondary"
                              href={`/api/runs/${currentRun.id}/history`}
                              download
                            >
                              Exact history
                            </a>
                          )}
                        </div>
                      )}
                    </section>
                    {result && (
                      <>
                        <div className="section-heading result-graph-heading">
                          <h2>Estimated structure</h2>
                          <div className="segmented">
                            <button
                              className={resultView === "dag" ? "selected" : ""}
                              onClick={() => setResultView("dag")}
                            >
                              DAG
                            </button>
                            <button
                              className={
                                resultView === "cpdag" ? "selected" : ""
                              }
                              onClick={() => setResultView("cpdag")}
                            >
                              Partially directed
                            </button>
                          </div>
                        </div>
                        <GraphEditor
                          key={`${currentRun.id}-${resultView}`}
                          graph={result.graph}
                          onChange={() => {}}
                          onError={fail}
                          readOnly
                          compact
                          title={
                            resultView === "dag"
                              ? "Representative DAG"
                              : "Equivalence structure"
                          }
                          displayEdges={
                            resultView === "cpdag"
                              ? result.cpdag_edges
                              : undefined
                          }
                        />
                        <p className="muted small result-footnote">
                          BIC is evaluated on the representative DAG and its
                          partition. Undirected edges express unresolved
                          orientations in the returned equivalence structure.
                        </p>
                      </>
                    )}
                    {runs.length > 0 && (
                      <section className="run-history">
                        <div className="section-heading">
                          <h2>Run history</h2>
                          <span className="muted small">
                            Inputs are saved with every run
                          </span>
                        </div>
                        {runs.map((run) => (
                          <button
                            key={run.id}
                            className={`history-row ${currentRun.id === run.id ? "selected" : ""}`}
                            onClick={() => {
                              setSelectedRun(run.id);
                              setResultView("dag");
                            }}
                          >
                            <span className="history-icon">
                              <Activity size={16} />
                            </span>
                            <span>
                              <strong>{run.name}</strong>
                              <small>
                                {dateLabel(run.created_at)} · seed{" "}
                                {run.request.settings.seed}
                              </small>
                            </span>
                            <span className={`history-state ${run.status}`}>
                              {run.status}
                            </span>
                            <strong className="history-score">
                              {formatScore(run.result?.score)}
                            </strong>
                            <ChevronRight size={14} />
                          </button>
                        ))}
                      </section>
                    )}
                  </>
                )}
              </div>
            </div>
          )}

          {page === "experiments" && (
            <>
              <div className="section-heading">
                <div>
                  <h2>Saved on this computer</h2>
                  <span className="muted small">
                    Named experiments preserve both editors and their settings.
                  </span>
                </div>
                <div className="section-actions">
                  <button
                    className="secondary"
                    onClick={() => experimentInput.current?.click()}
                  >
                    <Upload size={14} />
                    Import experiment
                  </button>
                  <button
                    className="primary"
                    onClick={() => {
                      if (defaultWorkspace.current) {
                        const next = clone(defaultWorkspace.current);
                        const sample = dataset || datasets[0];
                        next.dataset_id = sample?.id ?? null;
                        next.candidate_graph = sample
                          ? emptyGraph(sample)
                          : { ...clone(next.source_graph), edges: [] };
                        setWorkspace(next);
                        setSavedId(null);
                        setSavedSnapshot("");
                        setPage("data");
                      }
                    }}
                  >
                    <Plus size={15} />
                    New experiment
                  </button>
                </div>
              </div>
              {experiments.length ? (
                <div className="experiment-grid">
                  {experiments.map((experiment) => (
                    <button
                      key={experiment.id}
                      className="experiment-card"
                      onClick={() => openExperiment(experiment)}
                    >
                      <div className="experiment-icon">
                        <GitBranch size={22} />
                        <ArrowUpRight size={17} />
                      </div>
                      <h3>{experiment.workspace.name}</h3>
                      <p>
                        {experiment.workspace.candidate_graph.nodes.length}{" "}
                        variables ·{" "}
                        {experiment.workspace.candidate_graph.edges.length}{" "}
                        candidate edges
                      </p>
                      <div>
                        <span>{dateLabel(experiment.updated_at)}</span>
                        <ChevronRight size={15} />
                      </div>
                    </button>
                  ))}
                </div>
              ) : (
                <div className="experiments-empty">
                  <FolderOpen size={34} />
                  <h2>A home for your experiments</h2>
                  <p>
                    Save your current work to return to it later, or import an
                    exported experiment.
                  </p>
                  <button className="primary" onClick={save} disabled={!!busy}>
                    <Save size={15} />
                    Save current experiment
                  </button>
                </div>
              )}
              <div className="info-box experiment-info">
                <FileJson size={18} />
                <span>
                  Exported experiments include the observations, both models,
                  and algorithm settings, so they can be moved to another
                  computer.
                </span>
              </div>
            </>
          )}
        </main>
        <footer className="app-footer">
          <span>
            <span className="status-dot" />
            Python computation · local storage
          </span>
          <span>Built on your DAGdiscovery algorithms</span>
        </footer>
      </div>
      <input
        ref={csvInput}
        aria-label="Choose CSV file"
        className="hidden-input"
        type="file"
        accept=".csv,text/csv"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) void importCsv(file);
          e.target.value = "";
        }}
      />
      <input
        ref={experimentInput}
        aria-label="Choose experiment file"
        className="hidden-input"
        type="file"
        accept=".json,application/json"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) void importExperiment(file);
          e.target.value = "";
        }}
      />
      {toast && (
        <div
          role={toast.error ? "alert" : "status"}
          className={`toast ${toast.error ? "error" : ""}`}
        >
          {toast.error ? <CircleHelp size={18} /> : <Check size={18} />}
          <span>{toast.message}</span>
          <button
            aria-label="Dismiss notification"
            onClick={() => setToast(null)}
          >
            <X size={15} />
          </button>
        </div>
      )}
      {help && (
        <div className="modal-backdrop" onClick={() => setHelp(false)}>
          <section
            className="help-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="help-title"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="modal-close"
              aria-label="Close about"
              onClick={() => setHelp(false)}
            >
              <X size={18} />
            </button>
            <span className="brand-symbol">
              <GitBranch />
            </span>
            <h2 id="help-title">A workbench for your DAGs</h2>
            <p>
              Generate linear Gaussian observations, edit candidate structures,
              and run your own discovery algorithms.
            </p>
            <h3>Reading the score</h3>
            <p>
              The counter matches the normalized Gaussian BIC in DAGdiscovery.
              Higher is better for models fitted to the same observations with
              the same centering convention. Color groups constrain error
              variances to be equal.
            </p>
            <h3>Reproducible experiments</h3>
            <p>
              Each run stores its seed, dataset fingerprint, graph, partition,
              settings, and code revision. MCMC MAP mode reports model visits
              separately from BIC; it does not include burn-in or thinning
              controls.
            </p>
            <h3>Local by design</h3>
            <p>
              The interface and worker run on this computer. Closing the
              application stops active workers. Saved experiments and completed
              runs remain on disk.
            </p>
            <div className="help-source">
              <GitBranch size={15} />
              <a
                href="https://github.com/TheHiddenSquid/DAGdiscovery"
                target="_blank"
                rel="noreferrer"
              >
                Original research repository
                <ArrowUpRight size={13} />
              </a>
            </div>
          </section>
        </div>
      )}
      {sampleManager && (
        <SampleManager
          samples={allSamples}
          activeId={workspace.dataset_id}
          busy={!!busy}
          runningSampleId={activeRun?.request.dataset_id}
          onClose={() => setSampleManager(false)}
          onDelete={(sample) => void deleteSample(sample)}
          onRestore={(sample) => void restoreSample(sample)}
        />
      )}
    </div>
  );
}
