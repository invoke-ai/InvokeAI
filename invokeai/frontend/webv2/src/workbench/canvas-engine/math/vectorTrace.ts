import type { Vec2 } from '@workbench/canvas-engine/types';

import { polygonBounds } from '@workbench/canvas-engine/freehand';

/** v6 taper profile: smoothstep pressure over up to half the path at either end. */
export const buildTaperedTracePoints = (
  coordinates: Vec2[],
  strokeWidth: number,
  taper = 100
): [number, number, number][] => {
  const distances = coordinates.map(() => 0);
  for (let i = 1; i < coordinates.length; i++) {
    distances[i] =
      distances[i - 1]! +
      Math.hypot(coordinates[i]!.x - coordinates[i - 1]!.x, coordinates[i]!.y - coordinates[i - 1]!.y);
  }
  const total = distances.at(-1) ?? 0;
  const length = Math.min(total / 2, Math.max(0, (strokeWidth * 6 * taper) / 100));
  return coordinates.map((point, i) => {
    const t = length === 0 ? 1 : Math.min(1, Math.min(distances[i]!, total - distances[i]!) / length);
    return [point.x, point.y, t * t * (3 - 2 * t)];
  });
};

/** Shared tension geometry for drawing and conservative convex-hull raster bounds. */
const traceLineControls = (points: Vec2[]) => {
  const controls: { before: Vec2; point: Vec2; after: Vec2 }[] = [];
  for (let i = 1; i < points.length - 1; i++) {
    const a = points[i - 1]!,
      b = points[i]!,
      c = points[i + 1]!;
    const d0 = Math.hypot(b.x - a.x, b.y - a.y),
      d1 = Math.hypot(c.x - b.x, c.y - b.y);
    if (d0 + d1 === 0) {
      continue;
    }
    const fa = (0.3 * d0) / (d0 + d1),
      fb = (0.3 * d1) / (d0 + d1);
    controls.push({
      before: { x: b.x - fa * (c.x - a.x), y: b.y - fa * (c.y - a.y) },
      point: b,
      after: { x: b.x + fb * (c.x - a.x), y: b.y + fb * (c.y - a.y) },
    });
  }
  return controls;
};

export const getTraceLineBounds = (points: Vec2[]) =>
  polygonBounds([...points, ...traceLineControls(points).flatMap((control) => [control.before, control.after])]);

/** Konva.Line's open tension-0.3 geometry, used by the v6 brush-line renderer. */
export const buildTraceLinePath = (
  points: Vec2[],
  createPath: (data?: string) => Path2D = () => new Path2D()
): Path2D => {
  const path = createPath();
  const first = points[0];
  if (!first) {
    return path;
  }
  path.moveTo(first.x, first.y);
  const controls = traceLineControls(points);
  const head = controls[0];
  if (!head) {
    for (const point of points.slice(1)) {
      path.lineTo(point.x, point.y);
    }
    return path;
  }
  path.quadraticCurveTo(head.before.x, head.before.y, head.point.x, head.point.y);
  for (let i = 1; i < controls.length; i++) {
    const previous = controls[i - 1]!,
      current = controls[i]!;
    path.bezierCurveTo(
      previous.after.x,
      previous.after.y,
      current.before.x,
      current.before.y,
      current.point.x,
      current.point.y
    );
  }
  const tail = controls.at(-1)!,
    last = points.at(-1)!;
  path.quadraticCurveTo(tail.after.x, tail.after.y, last.x, last.y);
  return path;
};

/** Same Q/T outline as v6 getSvgPathFromStroke (including its two-decimal coordinates). */
export const buildTraceOutlinePath = (
  points: number[][],
  createPath: (data?: string) => Path2D = (data) => (data ? new Path2D(data) : new Path2D())
): Path2D => {
  if (points.length < 4) {
    return createPath();
  }
  const value = (n: number) => n.toFixed(2);
  const midpoint = (a: number[], b: number[]) => `${value((a[0]! + b[0]!) / 2)},${value((a[1]! + b[1]!) / 2)}`;
  const a = points[0]!,
    b = points[1]!,
    c = points[2]!;
  let data = `M${value(a[0]!)},${value(a[1]!)} Q${value(b[0]!)},${value(b[1]!)} ${midpoint(b, c)} T`;
  for (let i = 2; i < points.length - 1; i++) {
    data += `${midpoint(points[i]!, points[i + 1]!)} `;
  }
  return createPath(`${data}Z`);
};
