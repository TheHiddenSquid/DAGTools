import { useState } from "react";
import { Dices, X } from "lucide-react";
import { api } from "./types";
import type { Graph } from "./types";

type Props = {
  graph: Graph;
  source: boolean;
  onApply: (graph: Graph) => void;
  onClose: () => void;
};
export function RandomGraphMenu({ graph, source, onApply, onClose }: Props) {
  const [mode, setMode] = useState("dag");
  const [values, setValues] = useState({
    p: graph.nodes.length,
    density: 0.5,
    num_colors: Math.min(3, graph.nodes.length),
    lambda_min: 0.25,
    lambda_max: 1,
    omega_min: 0.2,
    omega_max: 2,
    seed: 42,
  });
  const [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const fields = [
    ...(mode === "dag"
      ? [
          ["p", "Variables · p", 1, 100, 1],
          ["density", "Edge probability · density", 0, 1, 0.05],
          ["lambda_min", "Min |coefficient| · λ min", 0, undefined, 0.05],
          ["lambda_max", "Max |coefficient| · λ max", 0, undefined, 0.05],
        ]
      : []),
    ["num_colors", "Available groups · num_colors", 1, values.p, 1],
    ["omega_min", "Min noise variance · ω min", 0.000001, undefined, 0.1],
    ["omega_max", "Max noise variance · ω max", 0.000001, undefined, 0.1],
    ["seed", "Random seed", 0, 4294967295, 1],
  ] as [keyof typeof values, string, number, number | undefined, number][];
  async function generate(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const { p, density, lambda_min, lambda_max, ...partition } = values;
      const body =
        mode === "dag"
          ? { ...values, graph: source ? null : graph }
          : { ...partition, graph };
      const next = await api<Graph>(
        mode === "dag" ? "/graphs/random" : "/graphs/random-partition",
        {
          method: "POST",
          body: JSON.stringify(body),
        },
      );
      onApply(next);
      onClose();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <div
      className="modal-backdrop"
      onClick={() => !busy && onClose()}
      onKeyDown={(event) => {
        if (event.key === "Escape" && !busy) onClose();
      }}
    >
      <section
        className="sample-modal random-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="random-title"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          className="modal-close"
          aria-label="Close random generator"
          onClick={onClose}
          disabled={busy}
        >
          <X size={18} />
        </button>
        <div className="eyebrow">MODEL GENERATOR</div>
        <h2 id="random-title">
          <Dices size={22} /> Create a random model
        </h2>
        <form onSubmit={generate}>
          <label>
            Generate
            <select
              autoFocus
              value={mode}
              onChange={(e) => {
                setMode(e.target.value);
                setValues((v) => ({
                  ...v,
                  p: graph.nodes.length,
                  num_colors: Math.min(v.num_colors, graph.nodes.length),
                }));
                setError("");
              }}
            >
              <option value="dag">Random DAG + partition</option>
              <option value="partition">Random partition only</option>
            </select>
          </label>
          <p className="muted small">
            {mode === "dag"
              ? "Replaces this DAG. Each pair in a random causal order receives an edge with the chosen probability; coefficients have random signs."
              : "Keeps the connections and positions; replaces variance groups and noise variances."}
          </p>
          {!source && mode === "dag" && (
            <p className="small">
              Variables stay matched to the active sample.
            </p>
          )}
          <div className="random-fields">
            {fields.map(([key, label, min, max, step]) => (
              <label key={key}>
                {label}
                <input
                  type="number"
                  required
                  min={min}
                  max={max}
                  step={step < 1 ? "any" : step}
                  disabled={key === "p" && !source}
                  value={values[key]}
                  onChange={(e) =>
                    setValues((v) => ({
                      ...v,
                      [key]: Number(e.target.value),
                      ...(key === "p"
                        ? {
                            num_colors: Math.max(
                              1,
                              Math.min(v.num_colors, Number(e.target.value)),
                            ),
                          }
                        : {}),
                    }))
                  }
                />
              </label>
            ))}
          </div>
          <p className="muted small">
            As in your existing generator, nodes choose groups independently, so
            some groups may be empty. Coefficient magnitudes and group variances
            are uniform within their bounds. The same seed repeats the same
            model. Creating a model does not change any saved sample.
          </p>
          {error && (
            <p className="error-banner" role="alert">
              {error}
            </p>
          )}
          <div className="random-actions">
            <button
              type="button"
              className="secondary"
              disabled={busy}
              onClick={onClose}
            >
              Cancel
            </button>
            <button className="primary" disabled={busy} type="submit">
              {busy
                ? "Generating…"
                : mode === "dag"
                  ? "Create random DAG"
                  : "Randomize partition"}
            </button>
          </div>
        </form>
      </section>
    </div>
  );
}
