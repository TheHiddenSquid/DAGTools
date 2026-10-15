import test from "node:test";
import assert from "node:assert/strict";
import { circularNodes } from "../.geometry-tests/graphLayout.js";

test("circular arrangement preserves variables with distinct noncollinear positions", () => {
  for (const count of [1, 2, 5, 20, 100]) {
    const input = Array.from({ length: count }, (_, i) => ({
      id: `x${i}`,
      x: 0,
      y: 0,
      group: "1",
    }));
    const result = circularNodes(input);
    assert.deepEqual(result, circularNodes(input));
    assert.equal(input[0].x, 0);
    for (let i = 0; i < count; i++) {
      assert.equal(result[i].id, input[i].id);
      for (let j = i + 1; j < count; j++) {
        const a = result[i],
          b = result[j],
          dx = b.x - a.x,
          dy = b.y - a.y,
          length = dx * dx + dy * dy;
        assert.ok(Math.sqrt(length) > 88);
        for (let k = 0; k < count; k++) {
          if (k === i || k === j) continue;
          const c = result[k],
            t = ((c.x - a.x) * dx + (c.y - a.y) * dy) / length;
          if (t <= 0 || t >= 1) continue;
          const distance = Math.hypot(c.x - a.x - t * dx, c.y - a.y - t * dy);
          // Distinct vertices of a circle cannot lie along another edge.
          assert.ok(distance > 1e-6);
        }
      }
    }
  }
});
