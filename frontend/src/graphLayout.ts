/** Stable circular placement: distinct vertices cannot hide collinear edges. */
export function circularNodes<T extends { x: number; y: number }>(
  nodes: T[],
): T[] {
  const radius = Math.max(150, (nodes.length * 120) / (2 * Math.PI));
  return nodes.map((node, i) => ({
    ...node,
    x:
      radius +
      radius * Math.cos(-Math.PI / 2 + (2 * Math.PI * i) / nodes.length),
    y:
      radius +
      radius * Math.sin(-Math.PI / 2 + (2 * Math.PI * i) / nodes.length),
  }));
}
