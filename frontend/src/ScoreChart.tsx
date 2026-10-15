import type { TracePoint } from "./types";
export function ScoreChart({ trace }: { trace: TracePoint[] }) {
  if (trace.length < 2)
    return (
      <div className="chart-empty">
        The score trace will appear as the search progresses.
      </div>
    );
  const values = trace.flatMap((p) => [p.score, p.best_score]);
  const min = Math.min(...values),
    max = Math.max(...values),
    range = max - min || 1;
  const last = Math.max(1, ...trace.map((p) => p.iteration));
  const points = (best: boolean) =>
    trace
      .map(
        (p) =>
          `${48 + (p.iteration / last) * 700},${125 - (((best ? p.best_score : p.score) - min) / range) * 96}`,
      )
      .join(" ");
  return (
    <div className="score-chart">
      <svg
        viewBox="0 0 780 157"
        role="img"
        aria-label="Current and best BIC score over search iterations"
      >
        <defs>
          <linearGradient id="chartFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#659982" stopOpacity="0.12" />
            <stop offset="100%" stopColor="#659982" stopOpacity="0" />
          </linearGradient>
        </defs>
        {[0, 0.5, 1].map((v) => (
          <g key={v}>
            <line
              x1="48"
              x2="750"
              y1={125 - v * 96}
              y2={125 - v * 96}
              stroke="#e9eee9"
              strokeDasharray="3 4"
            />
            <text x="37" y={129 - v * 96} textAnchor="end">
              {(min + v * range).toFixed(2)}
            </text>
          </g>
        ))}
        <polygon
          points={`48,125 ${points(true)} 748,125`}
          fill="url(#chartFill)"
        />
        <polyline
          points={points(false)}
          fill="none"
          stroke="#b8c6bd"
          strokeWidth="1.5"
        />
        <polyline
          points={points(true)}
          fill="none"
          stroke="#367362"
          strokeWidth="2"
        />
        <text x="48" y="149">
          0
        </text>
        <text x="748" y="149" textAnchor="end">
          {last.toLocaleString()} iterations
        </text>
      </svg>
    </div>
  );
}
