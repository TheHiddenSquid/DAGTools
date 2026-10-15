import {
  memo,
  useCallback,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  applyNodeChanges,
  Background,
  BackgroundVariant,
  BaseEdge,
  ConnectionLineType,
  Controls,
  Handle,
  MarkerType,
  Position,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  useInternalNode,
  getStraightPath,
} from "@xyflow/react";
import type {
  Connection as FlowConnection,
  NodeProps,
  Node as FlowNode,
  NodeChange,
  EdgeProps,
} from "@xyflow/react";
import {
  Dices,
  ArrowLeftRight,
  ArrowUpRight,
  CornerDownRight,
  GitBranch,
  LayoutGrid,
  MousePointer2,
  Plus,
  Redo2,
  Trash2,
  Undo2,
} from "lucide-react";
import type { ReactNode } from "react";
import type { Graph } from "./types";
import { clone, groupColor } from "./types";
import { circularNodes } from "./graphLayout";
import { RandomGraphMenu } from "./RandomGraphMenu";
import { circleSegment } from "./edgeGeometry";

type VariableFlowNode = FlowNode<
  { label: string; group: string; color: string; sameGroup: boolean },
  "variable"
>;
const VariableNode = memo(function VariableNode({
  data,
  selected,
}: NodeProps<VariableFlowNode>) {
  return (
    <div
      className={`variable-node ${selected ? "is-selected" : data.sameGroup ? "is-group-peer" : ""}`}
      style={{ "--node-color": data.color } as React.CSSProperties}
    >
      {(selected || data.sameGroup) && (
        <span
          className={`node-selection-badge ${selected ? "selected-badge" : ""}`}
          title={
            selected
              ? "Selected variable"
              : "Same variance group as selected variable"
          }
          aria-label={selected ? "Selected variable" : "Same variance group"}
        >
          {selected ? "✓" : "●"}
        </span>
      )}
      <Handle type="target" position={Position.Left} />
      <span title={data.label}>
        <strong>{data.label}</strong>
        <small>Group {data.group}</small>
      </span>
      <Handle type="source" position={Position.Right} />
    </div>
  );
});
const nodeTypes = { variable: VariableNode };

function CircleEdge(props: EdgeProps) {
  const source = useInternalNode(props.source);
  const target = useInternalNode(props.target);
  if (!source || !target) return null;
  const segment = circleSegment(
    {
      ...source.internals.positionAbsolute,
      width: source.measured.width ?? 88,
      height: source.measured.height ?? 88,
    },
    {
      ...target.internals.positionAbsolute,
      width: target.measured.width ?? 88,
      height: target.measured.height ?? 88,
    },
  );
  if (!segment) return null;
  const [path, labelX, labelY] = getStraightPath(segment);
  return (
    <BaseEdge
      id={props.id}
      path={path}
      labelX={labelX}
      labelY={labelY}
      markerEnd={props.markerEnd}
      style={props.style}
      label={props.label}
      labelStyle={props.labelStyle}
      labelBgStyle={props.labelBgStyle}
      labelBgPadding={props.labelBgPadding}
      interactionWidth={20}
    />
  );
}
const edgeTypes = { circle: CircleEdge };

function wouldCycle(graph: Graph, source: string, target: string) {
  const pending = [target],
    visited = new Set<string>();
  while (pending.length) {
    const id = pending.pop()!;
    if (id === source) return true;
    if (visited.has(id)) continue;
    visited.add(id);
    for (const edge of graph.edges)
      if (edge.source === id) pending.push(edge.target);
  }
  return false;
}

type Props = {
  graph: Graph;
  onChange: (graph: Graph) => void;
  source?: boolean;
  readOnly?: boolean;
  compact?: boolean;
  title?: string;
  footer?: ReactNode;
  onError: (message: string) => void;
  displayEdges?: { source: string; target: string; directed: boolean }[];
};
function Editor({
  graph,
  onChange,
  source = false,
  readOnly = false,
  compact = false,
  title,
  footer,
  onError,
  displayEdges,
}: Props) {
  const flow = useReactFlow<VariableFlowNode>();
  const [randomOpen, setRandomOpen] = useState(false);
  const [selected, setSelected] = useState<{
    kind: "node" | "edge";
    id: string;
  } | null>(null);
  const [past, setPast] = useState<Graph[]>([]),
    [future, setFuture] = useState<Graph[]>([]);
  const dragStart = useRef<Graph | null>(null);
  const groups = useMemo(
    () => [...new Set(graph.nodes.map((n) => n.group))],
    [graph.nodes],
  );
  const commit = useCallback(
    (next: Graph) => {
      setPast((p) => [...p.slice(-39), clone(graph)]);
      setFuture([]);
      onChange(next);
    },
    [graph, onChange],
  );
  const node =
    selected?.kind === "node"
      ? graph.nodes.find((n) => n.id === selected.id)
      : undefined;
  const edge =
    selected?.kind === "edge"
      ? graph.edges.find((e) => e.id === selected.id)
      : undefined;
  const modelNodes = useMemo(
    () =>
      graph.nodes.map((n) => ({
        id: n.id,
        type: "variable" as const,
        width: 88,
        height: 88,
        position: { x: n.x, y: n.y },
        selected: selected?.kind === "node" && selected.id === n.id,
        deletable: source && graph.nodes.length > 1,
        data: {
          label: n.label,
          group: n.group,
          color: groupColor(n.group, groups),
          sameGroup: !!node && n.id !== node.id && n.group === node.group,
        },
      })),
    [graph.nodes, selected, source, groups, node],
  );
  const [nodes, setNodes] = useState<VariableFlowNode[]>(modelNodes);
  useLayoutEffect(() => {
    if (dragStart.current) return;
    setNodes((current) =>
      modelNodes.map((node) => {
        const previous = current.find((n) => n.id === node.id);
        return { ...previous, ...node };
      }),
    );
  }, [modelNodes]);
  const edges = useMemo(
    () =>
      (displayEdges || graph.edges.map((e) => ({ ...e, directed: true }))).map(
        (e, i) => ({
          id: "id" in e ? String(e.id) : `result-${i}`,
          source: e.source,
          target: e.target,
          markerEnd: e.directed
            ? {
                type: MarkerType.ArrowClosed,
                width: 18,
                height: 18,
                color: "#637b70",
              }
            : undefined,
          selected:
            selected?.kind === "edge" &&
            selected.id === ("id" in e ? e.id : `result-${i}`),
          label:
            source && "weight" in e ? Number(e.weight).toFixed(2) : undefined,
          style: { stroke: "#637b70", strokeWidth: 1.8 },
          labelStyle: { fill: "#69766e", fontSize: 10 },
          labelBgStyle: { fill: "#fafbf9" },
          labelBgPadding: [5, 3] as [number, number],
          type: "circle",
        }),
      ),
    [displayEdges, graph.edges, selected, source],
  );
  function connect(connection: FlowConnection) {
    const { source: a, target: b } = connection;
    if (a === b || graph.edges.some((e) => e.source === a && e.target === b))
      return;
    if (wouldCycle(graph, a, b)) {
      onError(
        "That connection would create a cycle. DAGs must remain acyclic.",
      );
      return;
    }
    commit({
      ...graph,
      edges: [
        ...graph.edges,
        { id: `${a}-${b}`, source: a, target: b, weight: 0.7 },
      ],
    });
  }
  function changes(changes: NodeChange<VariableFlowNode>[]) {
    const removed = changes.filter((c) => c.type === "remove").map((c) => c.id);
    if (removed.length && source && graph.nodes.length > removed.length) {
      commit({
        ...graph,
        nodes: graph.nodes.filter((n) => !removed.includes(n.id)),
        edges: graph.edges.filter(
          (e) => !removed.includes(e.source) && !removed.includes(e.target),
        ),
      });
      setSelected(null);
      return;
    }
    // Preserve React Flow's dimensions, selection and dragging state locally.
    // Persist positions once on release, avoiding full app renders and autosave
    // while the pointer is moving.
    setNodes((current) =>
      applyNodeChanges(
        changes.filter((c) => c.type !== "remove"),
        current,
      ),
    );
    const positions = changes.filter((c) => c.type === "position");
    if (!readOnly && !dragStart.current && positions.length) {
      // Keyboard arrow moves do not emit pointer drag callbacks.
      commit({
        ...graph,
        nodes: graph.nodes.map((n) => {
          const change = positions.find((c) => c.id === n.id);
          return change?.position
            ? { ...n, x: change.position.x, y: change.position.y }
            : n;
        }),
      });
    }
  }

  function updateNode(update: Partial<(typeof graph.nodes)[number]>) {
    if (node)
      commit({
        ...graph,
        nodes: graph.nodes.map((n) =>
          n.id === node.id ? { ...n, ...update } : n,
        ),
      });
  }
  function remove() {
    if (node && source && graph.nodes.length > 1) {
      commit({
        ...graph,
        nodes: graph.nodes.filter((n) => n.id !== node.id),
        edges: graph.edges.filter(
          (e) => e.source !== node.id && e.target !== node.id,
        ),
      });
      setSelected(null);
    }
    if (edge) {
      commit({ ...graph, edges: graph.edges.filter((e) => e.id !== edge.id) });
      setSelected(null);
    }
  }
  function addNode() {
    let index = graph.nodes.length + 1;
    while (
      graph.nodes.some((n) => n.id === `x${index}` || n.label === `X${index}`)
    )
      index++;
    if (graph.nodes.length >= 100) {
      onError("The first version supports up to 100 variables.");
      return;
    }
    const next = {
      id: `x${index}`,
      label: `X${index}`,
      group: "1",
      x: 100 + (graph.nodes.length % 4) * 190,
      y: 80 + Math.floor(graph.nodes.length / 4) * 140,
    };
    commit({ ...graph, nodes: [...graph.nodes, next] });
    setSelected({ kind: "node", id: next.id });
  }
  function layout() {
    const next = { ...graph, nodes: circularNodes(graph.nodes) };
    if (readOnly) onChange(next);
    else commit(next);
    requestAnimationFrame(() => flow.fitView({ padding: 0.2, duration: 300 }));
  }
  return (
    <div className={`editor-grid ${compact ? "compact" : ""}`}>
      {randomOpen && (
        <RandomGraphMenu
          graph={graph}
          source={source}
          onClose={() => setRandomOpen(false)}
          onApply={(next) => {
            commit(next);
            setSelected(null);
            requestAnimationFrame(() =>
              flow.fitView({ padding: 0.23, duration: 300 }),
            );
          }}
        />
      )}
      <section className="canvas-card">
        <div className="canvas-toolbar">
          <div className="canvas-title">
            <GitBranch size={15} />
            <strong>{title || "Candidate DAG"}</strong>
            <span className="subtle-tag">{graph.nodes.length} variables</span>
          </div>
          <div className="toolbar-actions">
            {!readOnly && (
              <>
                <button
                  aria-label="Undo graph edit"
                  title="Undo"
                  disabled={!past.length}
                  onClick={() => {
                    const previous = past.at(-1)!;
                    setPast(past.slice(0, -1));
                    setFuture([clone(graph), ...future]);
                    onChange(previous);
                    setSelected(null);
                  }}
                >
                  <Undo2 size={16} />
                </button>
                <button
                  aria-label="Redo graph edit"
                  title="Redo"
                  disabled={!future.length}
                  onClick={() => {
                    setPast([...past, clone(graph)]);
                    onChange(future[0]);
                    setFuture(future.slice(1));
                  }}
                >
                  <Redo2 size={16} />
                </button>
                <span className="toolbar-divider" />
              </>
            )}
            {!readOnly && (
              <button
                aria-label="Arrange graph"
                title="Arrange graph in a circle"
                onClick={layout}
              >
                <LayoutGrid size={16} />
              </button>
            )}
            {!readOnly && (
              <button className="small-add" onClick={() => setRandomOpen(true)}>
                <Dices size={15} /> Random
              </button>
            )}
            {source && !readOnly && (
              <button className="small-add" onClick={addNode}>
                <Plus size={15} />
                Variable
              </button>
            )}
          </div>
        </div>
        <div className="flow-surface">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            connectionLineType={ConnectionLineType.Straight}
            fitView
            fitViewOptions={{ padding: 0.23, maxZoom: 1.1 }}
            minZoom={0.15}
            maxZoom={2}
            nodesDraggable={!readOnly}
            nodesConnectable={!readOnly}
            elementsSelectable={!readOnly}
            deleteKeyCode={readOnly ? null : ["Backspace", "Delete"]}
            onConnect={connect}
            onNodesChange={changes}
            onNodeDragStart={() => {
              dragStart.current = clone(graph);
            }}
            onNodeDragStop={() => {
              const previous = dragStart.current;
              if (previous) {
                setPast((p) => [...p.slice(-39), previous]);
                setFuture([]);
                const positions = new Map(
                  flow.getNodes().map((n) => [n.id, n.position]),
                );
                dragStart.current = null;
                onChange({
                  ...graph,
                  nodes: graph.nodes.map((n) => {
                    const position = positions.get(n.id);
                    return position
                      ? { ...n, x: position.x, y: position.y }
                      : n;
                  }),
                });
              }
            }}
            onNodeClick={(_, n) => setSelected({ kind: "node", id: n.id })}
            onEdgeClick={(_, e) => setSelected({ kind: "edge", id: e.id })}
            onPaneClick={() => setSelected(null)}
            onEdgesDelete={(deleted) => {
              if (!readOnly)
                commit({
                  ...graph,
                  edges: graph.edges.filter(
                    (e) => !deleted.some((d) => d.id === e.id),
                  ),
                });
            }}
          >
            <Background
              variant={BackgroundVariant.Dots}
              color="#cfd9d2"
              gap={22}
              size={1}
            />
            <Controls showInteractive={false} />
          </ReactFlow>
          {!readOnly && (
            <div className="canvas-hint">
              <MousePointer2 size={12} />
              {source
                ? "Draw connections to define your model"
                : "Connect variables to explore a structure"}
              <span>Drag from a node’s right port</span>
            </div>
          )}
        </div>
        <div className="canvas-legend">
          <span className="legend-label">VARIANCE GROUPS</span>
          {groups.map((g) => (
            <span
              key={g}
              className={node?.group === g ? "selected-group-label" : ""}
            >
              <i style={{ background: groupColor(g, groups) }} />
              Group {g}
            </span>
          ))}
          <span className="canvas-edge-count">
            {displayEdges?.length ?? graph.edges.length} edges
          </span>
        </div>
      </section>
      {!compact && (
        <aside className="inspector">
          <div className="panel-heading">
            <span className="eyebrow">MODEL DETAILS</span>
            <span className="dot" />
          </div>
          {node ? (
            <>
              <h3>
                {node.label}
                <span className="subtle-tag">Variable</span>
              </h3>
              {source ? (
                <label>
                  Name
                  <input
                    key={`${node.id}-label`}
                    defaultValue={node.label}
                    onBlur={(e) => {
                      const value = e.target.value.trim();
                      if (
                        !value ||
                        graph.nodes.some(
                          (n) => n.id !== node.id && n.label === value,
                        )
                      ) {
                        e.target.value = node.label;
                        onError("Choose a nonempty, unique variable name.");
                      } else if (value !== node.label)
                        updateNode({ label: value });
                    }}
                  />
                </label>
              ) : (
                <p className="muted small">
                  Variable names follow the active dataset.
                </p>
              )}
              <label>
                Variance group
                <select
                  value={node.group}
                  onChange={(e) => {
                    if (e.target.value === "__new") {
                      const group = String(
                        Math.max(0, ...groups.map((g) => Number(g) || 0)) + 1,
                      );
                      commit({
                        ...graph,
                        nodes: graph.nodes.map((n) =>
                          n.id === node.id ? { ...n, group } : n,
                        ),
                        variances: { ...graph.variances, [group]: 1 },
                      });
                    } else updateNode({ group: e.target.value });
                  }}
                >
                  {groups.map((g) => (
                    <option key={g} value={g}>
                      Group {g}
                    </option>
                  ))}
                  <option value="__new">+ New group</option>
                </select>
              </label>
              <label>
                Connect to
                <select
                  aria-label="Connect selected variable to"
                  value=""
                  onChange={(e) => {
                    if (e.target.value)
                      connect({
                        source: node.id,
                        target: e.target.value,
                        sourceHandle: null,
                        targetHandle: null,
                      });
                  }}
                >
                  <option value="">Choose a target variable…</option>
                  {graph.nodes
                    .filter(
                      (n) =>
                        n.id !== node.id &&
                        !graph.edges.some(
                          (e) => e.source === node.id && e.target === n.id,
                        ),
                    )
                    .map((n) => (
                      <option key={n.id} value={n.id}>
                        {n.label}
                      </option>
                    ))}
                </select>
              </label>
              {source && (
                <label>
                  Group noise variance
                  <input
                    type="number"
                    min="0.000001"
                    step="0.1"
                    value={graph.variances[node.group] ?? 1}
                    onChange={(e) => {
                      const v = Number(e.target.value);
                      if (v > 0)
                        commit({
                          ...graph,
                          variances: { ...graph.variances, [node.group]: v },
                        });
                    }}
                  />
                  <small>Shared by every node in group {node.group}.</small>
                </label>
              )}
              <div className="detail-row">
                <span>Same variance group</span>
                <strong>
                  {graph.nodes
                    .filter((n) => n.group === node.group && n.id !== node.id)
                    .map((n) => n.label)
                    .join(", ") || "No other variables"}
                </strong>
              </div>
              <div className="detail-row">
                <span>Parents</span>
                <strong>
                  {graph.edges
                    .filter((e) => e.target === node.id)
                    .map(
                      (e) => graph.nodes.find((n) => n.id === e.source)?.label,
                    )
                    .join(", ") || "None"}
                </strong>
              </div>
              {source && (
                <button
                  className="text-danger"
                  disabled={graph.nodes.length <= 1}
                  onClick={remove}
                >
                  <Trash2 size={14} />
                  Remove variable
                </button>
              )}
            </>
          ) : edge ? (
            <>
              <h3>
                Connection<span className="subtle-tag">Edge</span>
              </h3>
              <div className="edge-pair">
                {graph.nodes.find((n) => n.id === edge.source)?.label}
                <ArrowUpRight size={17} />
                {graph.nodes.find((n) => n.id === edge.target)?.label}
              </div>
              {source && (
                <label>
                  Coefficient
                  <input
                    type="number"
                    step="0.1"
                    value={edge.weight}
                    onChange={(e) => {
                      const v = Number(e.target.value);
                      if (Number.isFinite(v))
                        commit({
                          ...graph,
                          edges: graph.edges.map((item) =>
                            item.id === edge.id ? { ...item, weight: v } : item,
                          ),
                        });
                    }}
                  />
                </label>
              )}
              <button
                className="secondary full"
                onClick={() => {
                  const without = {
                    ...graph,
                    edges: graph.edges.filter((e) => e.id !== edge.id),
                  };
                  if (wouldCycle(without, edge.target, edge.source)) {
                    onError("Reversing this edge would create a cycle.");
                    return;
                  }
                  commit({
                    ...without,
                    edges: [
                      ...without.edges,
                      {
                        ...edge,
                        id: `${edge.target}-${edge.source}`,
                        source: edge.target,
                        target: edge.source,
                      },
                    ],
                  });
                  setSelected(null);
                }}
              >
                <ArrowLeftRight size={14} />
                Reverse direction
              </button>
              <button className="text-danger" onClick={remove}>
                <Trash2 size={14} />
                Remove edge
              </button>
            </>
          ) : (
            <>
              <div className="inspector-symbol">
                <MousePointer2 size={23} />
              </div>
              <h3>Make it your model</h3>
              <p className="muted small">
                Select a variable to edit its variance group, or select an edge
                to change the structure.
              </p>
              <div className="instruction">
                <CornerDownRight size={16} />
                <span>Nodes with the same color share an error variance.</span>
              </div>
            </>
          )}
          {footer && <div className="inspector-footer">{footer}</div>}
        </aside>
      )}
    </div>
  );
}
export function GraphEditor(props: Props) {
  return (
    <ReactFlowProvider>
      <Editor {...props} />
    </ReactFlowProvider>
  );
}
