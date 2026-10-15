type CircleBounds = { x: number; y: number; width: number; height: number };

/** A straight segment clipped to the two circle boundaries, in any direction. */
export function circleSegment(source: CircleBounds, target: CircleBounds) {
  const sourceX = source.x + source.width / 2;
  const sourceY = source.y + source.height / 2;
  const targetX = target.x + target.width / 2;
  const targetY = target.y + target.height / 2;
  const dx = targetX - sourceX;
  const dy = targetY - sourceY;
  const distance = Math.hypot(dx, dy);
  const sourceRadius = Math.min(source.width, source.height) / 2 + 2;
  const targetRadius = Math.min(target.width, target.height) / 2 + 2;
  // Overlapping circles have no visible segment; never draw a reversed arrow.
  if (distance <= sourceRadius + targetRadius) return null;
  return {
    sourceX: sourceX + (dx / distance) * sourceRadius,
    sourceY: sourceY + (dy / distance) * sourceRadius,
    targetX: targetX - (dx / distance) * targetRadius,
    targetY: targetY - (dy / distance) * targetRadius,
  };
}
