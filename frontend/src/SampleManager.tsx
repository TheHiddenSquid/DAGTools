import { Archive, Database, RotateCcw, Trash2, X } from "lucide-react";
import type { Dataset } from "./types";

type Props = {
  samples: Dataset[];
  activeId: string | null;
  busy: boolean;
  runningSampleId?: string;
  onClose: () => void;
  onDelete: (sample: Dataset) => void;
  onRestore: (sample: Dataset) => void;
  onPurge: (sample: Dataset) => void;
};

export function SampleManager({
  samples,
  activeId,
  busy,
  runningSampleId,
  onClose,
  onDelete,
  onRestore,
  onPurge,
}: Props) {
  const available = samples.filter((sample) => !sample.deleted_at);
  const deleted = samples.filter((sample) => sample.deleted_at);
  function row(sample: Dataset) {
    const isDeleted = !!sample.deleted_at;
    const running = runningSampleId === sample.id;
    return (
      <div className="sample-row" key={sample.id}>
        <span className="sample-row-icon">
          <Database size={18} />
        </span>
        <div>
          <strong>
            {sample.name}
            {!isDeleted && sample.id === activeId && (
              <span className="subtle-tag">Active</span>
            )}
          </strong>
          <small>
            {sample.n.toLocaleString()} observations · {sample.p} variables
            {running ? " · Used by active run" : ""}
          </small>
        </div>
        <button
          className={isDeleted ? "secondary" : "sample-delete"}
          disabled={busy || running}
          aria-label={`${isDeleted ? "Restore" : "Hide"} sample ${sample.name}`}
          title={
            running
              ? "Stop the active run before deleting its sample"
              : isDeleted
                ? "Restore sample"
                : "Hide sample from selector"
          }
          onClick={() => (isDeleted ? onRestore(sample) : onDelete(sample))}
        >
          {isDeleted ? (
            <>
              <RotateCcw size={14} />
              Restore
            </>
          ) : (
            <Archive size={16} />
          )}
        </button>
        <button
          className="sample-delete"
          disabled={busy || running}
          aria-label={`Permanently delete sample ${sample.name}`}
          title="Permanently delete sample and related runs"
          onClick={() => onPurge(sample)}
        >
          <Trash2 size={16} />
          <span className="small">Delete forever</span>
        </button>
      </div>
    );
  }
  return (
    <div
      className="modal-backdrop"
      onClick={onClose}
      onKeyDown={(event) => {
        if (event.key === "Escape") onClose();
      }}
    >
      <section
        className="sample-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="samples-title"
        onClick={(event) => event.stopPropagation()}
      >
        <button
          className="modal-close"
          autoFocus
          aria-label="Close sample manager"
          onClick={onClose}
        >
          <X size={18} />
        </button>
        <div className="eyebrow">YOUR OBSERVATIONS</div>
        <h2 id="samples-title">Manage samples</h2>
        <p>
          Hide samples from the selector and restore them later, or choose
          Delete forever to erase their observations and related run records.
        </p>
        <div className="sample-list">
          {available.length ? (
            available.map(row)
          ) : (
            <p className="samples-empty">
              No active samples. Generate, import, or restore a sample to
              continue scoring.
            </p>
          )}
        </div>
        {deleted.length > 0 && (
          <details className="deleted-samples" open>
            <summary>Hidden samples ({deleted.length})</summary>
            {deleted.map(row)}
          </details>
        )}
      </section>
    </div>
  );
}
