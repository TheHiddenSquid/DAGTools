import { circularNodes } from "./graphLayout";
export type Variable = {
  id: string;
  label: string;
  group: string;
  x: number;
  y: number;
};
export type Connection = {
  id: string;
  source: string;
  target: string;
  weight: number;
};
export type Graph = {
  nodes: Variable[];
  edges: Connection[];
  variances: Record<string, number>;
};
export type Dataset = {
  deleted_at?: string;
  id: string;
  name: string;
  n: number;
  p: number;
  columns: { id: string; label: string }[];
  created_at: string;
  sha256: string;
  preview: number[][];
  provenance: { type: string; seed?: number; graph?: Graph };
};
export type Settings = {
  seed: number;
  center: boolean;
  start_from_candidate: boolean;
  fixed_partition: boolean;
  num_iters: number;
  num_starts: number;
  mode: "bic" | "map";
  color_probability: number;
  moves: string[];
  refine_partition: boolean;
  target_bic: number | null;
  record_history: boolean;
  chunk_size: number;
  ges_phases: string[];
  ges_iterate: boolean;
};
export type Workspace = {
  schema_version: 1;
  name: string;
  dataset_id: string | null;
  source_graph: Graph;
  candidate_graph: Graph;
  algorithm: string;
  settings: Settings;
  center: boolean;
};
export type Experiment = {
  id: string;
  updated_at: string;
  workspace: Workspace;
};
export type Score = {
  score: number;
  higher_is_better: boolean;
  edges: number;
  groups: number;
  parameters: number;
  edge_penalty: number;
  blocks: { nodes: string[]; variance: number; contribution: number }[];
  sample_size: number;
};
export type TracePoint = {
  iteration: number;
  score: number;
  best_score: number;
};
export type Progress = {
  phase: string;
  iteration: number;
  restart?: number;
  total?: number;
  score?: number;
  best_score?: number;
  elapsed: number;
  trace: TracePoint[];
};
export type Run = {
  id: string;
  name: string;
  status: string;
  created_at: string;
  finished_at?: string;
  request: {
    dataset_id: string;
    graph: Graph;
    algorithm: string;
    settings: Settings;
  };
  progress?: Progress;
  error?: string;
  provenance: Record<string, unknown>;
  result?: {
    graph: Graph;
    cpdag_edges: { source: string; target: string; directed: boolean }[];
    score: number;
    score_details: Score;
    diagnostics: Record<string, number | boolean | null>;
    elapsed: number;
    trace: TracePoint[];
    history_available?: boolean;
  };
};
export type Field = {
  name: keyof Settings;
  label: string;
  type: string;
  hint?: string;
  min?: number;
  max?: number;
  step?: number;
  options?: { value: string; label: string }[];
  visible_if?: Partial<Settings>;
};
export type Algorithm = {
  id: string;
  name: string;
  description: string;
  source: string;
  progress_unit: string;
  fields: Field[];
};

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...init,
    headers: {
      ...(init?.body && !(init.body instanceof FormData)
        ? { "Content-Type": "application/json" }
        : {}),
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const error = await response
      .json()
      .catch(() => ({ detail: response.statusText }));
    throw new Error(
      typeof error.detail === "string"
        ? error.detail
        : error.detail?.map((item: { msg: string }) => item.msg).join("; ") ||
            "The request failed.",
    );
  }
  return response.json();
}
export const clone = <T>(value: T): T => structuredClone(value);
export const formatScore = (value: number | undefined) =>
  value === undefined ? "—" : value.toFixed(5);
export const colors = [
  "#5b8c77",
  "#8e7cbb",
  "#c49055",
  "#5d91b0",
  "#c57583",
  "#899a4e",
  "#688e98",
  "#b586b4",
];
export function groupColor(group: string, groups: string[]) {
  return colors[Math.max(0, groups.indexOf(group)) % colors.length];
}
export function emptyGraph(dataset: Dataset): Graph {
  return {
    nodes: circularNodes(
      dataset.columns.map((c, i) => ({
        ...c,
        group: String(i + 1),
        x: 60 + (i % 4) * 200,
        y: 80 + Math.floor(i / 4) * 140,
      })),
    ),
    edges: [],
    variances: Object.fromEntries(
      dataset.columns.map((_, i) => [String(i + 1), 1]),
    ),
  };
}
export function download(payload: unknown, name: string) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}
