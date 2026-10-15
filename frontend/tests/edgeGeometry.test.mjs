import { test } from "node:test";
import assert from "node:assert/strict";
import { circleSegment } from "../.geometry-tests/edgeGeometry.js";

for (const [name, x, y] of [
  ["right", 250, 0],
  ["left", -250, 0],
  ["down", 0, 250],
  ["up", 0, -250],
  ["diagonal", 250, -180],
]) {
  test(`straight edge is clipped to both circles facing ${name}`, () => {
    const source = { x: 0, y: 0, width: 88, height: 88 },
      target = { x, y, width: 88, height: 88 };
    const segment = circleSegment(source, target);
    assert.ok(segment);
    assert.ok(
      Math.abs(Math.hypot(segment.sourceX - 44, segment.sourceY - 44) - 46) <
        1e-9,
    );
    assert.ok(
      Math.abs(
        Math.hypot(segment.targetX - (x + 44), segment.targetY - (y + 44)) - 46,
      ) < 1e-9,
    );
    const cross =
      (segment.targetX - segment.sourceX) * y -
      (segment.targetY - segment.sourceY) * x;
    assert.ok(
      Math.abs(cross) < 1e-8,
      "segment follows the center-to-center line",
    );
    assert.ok(
      (segment.targetX - segment.sourceX) * x +
        (segment.targetY - segment.sourceY) * y >
        0,
      "arrow points toward target",
    );
  });
}
test("overlapping nodes do not produce backward arrows", () => {
  assert.equal(
    circleSegment(
      { x: 0, y: 0, width: 88, height: 88 },
      { x: 20, y: 20, width: 88, height: 88 },
    ),
    null,
  );
});
test("different circle sizes use their own radii", () => {
  const segment = circleSegment(
    { x: 0, y: 0, width: 88, height: 88 },
    { x: 250, y: -6, width: 100, height: 100 },
  );
  assert.equal(segment.sourceX, 90);
  assert.equal(segment.targetX, 248);
});
