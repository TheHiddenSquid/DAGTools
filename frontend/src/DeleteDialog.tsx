import { Trash2, X } from "lucide-react";
import type { DeletionPlan } from "./types";

export function DeleteDialog({
  plan,
  busy,
  onCancel,
  onConfirm,
}: {
  plan: DeletionPlan;
  busy: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const experiment = plan.kind === "workspace";
  return (
    <div
      className="modal-backdrop"
      onClick={() => !busy && onCancel()}
      onKeyDown={(event) => {
        if (event.key === "Escape" && !busy) onCancel();
      }}
    >
      <section
        className="sample-modal delete-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="delete-title"
        onClick={(event) => event.stopPropagation()}
      >
        <button
          className="modal-close"
          disabled={busy}
          aria-label="Cancel permanent deletion"
          onClick={onCancel}
        >
          <X size={18} />
        </button>
        <div className="eyebrow">PERMANENT DELETION</div>
        <h2 id="delete-title">
          Delete {experiment ? "experiment" : "sample"} forever?
        </h2>
        <p>
          <strong>{plan.name}</strong>
        </p>
        <p>
          {experiment
            ? "This erases the saved experiment and all of its algorithm run records, results, traces, and history files. Its data samples remain available to other experiments."
            : "This erases the observations, sample preview, and all algorithm run records that used this sample, including results and history files."}
        </p>
        <p>
          <strong>
            {plan.run_ids.length} run{" "}
            {plan.run_ids.length === 1 ? "record" : "records"} will be
            permanently removed.
          </strong>
        </p>
        {!experiment && plan.experiments.length > 0 && (
          <>
            <p>
              These experiments will keep their graphs and settings, with no
              active sample:
            </p>
            <ul>
              {plan.experiments.map((e) => (
                <li key={e.id}>{e.name}</li>
              ))}
            </ul>
          </>
        )}
        <p className="delete-warning">
          This cannot be undone. Any files you previously exported remain on
          disk.
        </p>
        {plan.running && (
          <p role="alert" className="error-banner">
            Stop the affected active run before deleting.
          </p>
        )}
        <div className="random-actions">
          <button
            autoFocus
            className="secondary"
            disabled={busy}
            onClick={onCancel}
          >
            Cancel
          </button>
          <button
            className="danger-button"
            disabled={busy || plan.running}
            onClick={onConfirm}
          >
            <Trash2 size={15} />
            {busy ? "Deleting…" : "Permanently delete"}
          </button>
        </div>
      </section>
    </div>
  );
}
