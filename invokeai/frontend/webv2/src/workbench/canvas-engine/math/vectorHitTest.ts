import type { CanvasBezierPathContract } from '../contracts';
import type { Mat2d, Rect, Vec2 } from '../types';

import { approximateBezierPath, getBezierPathBounds, getBezierPathHitSamplesPerSegment } from './bezierPath';
import { applyToPoint } from './mat2d';

type Entry = { key: string; bounds: Rect; anchors: Vec2[]; samples: Vec2[]; steps: number };
const MAX_CACHED_SAMPLES = 50_000;

/** Controller-owned cache: only immutable geometry + the current projection, with bounded retained samples. */
export class VectorHitTestCache {
  private entries = new Map<CanvasBezierPathContract, Entry>();
  private retained = 0;

  clear(): void { this.entries.clear(); this.retained = 0; }

  private entry(path: CanvasBezierPathContract, matrix: Mat2d, scale: number): Entry {
    const steps = getBezierPathHitSamplesPerSegment(scale);
    const key = `${matrix.a},${matrix.b},${matrix.c},${matrix.d},${matrix.e},${matrix.f},${steps}`;
    let entry = this.entries.get(path);
    if (entry?.key !== key) {
      if (entry) { this.retained -= entry.samples.length + entry.anchors.length; this.entries.delete(path); }
      const projected = path.points.map((p) => ({ anchor: applyToPoint(matrix, p.anchor),
        inHandle: p.inHandle ? applyToPoint(matrix, p.inHandle) : null,
        outHandle: p.outHandle ? applyToPoint(matrix, p.outHandle) : null }));
      entry = { key, steps, bounds: getBezierPathBounds(projected, path.isClosed), anchors: projected.map((p) => p.anchor), samples: [] };
      this.retained += entry.anchors.length;
      // Keep even offscreen bounds, but do not sample a distant curve just to reject it.
      this.entries.set(path, entry);
    }
    this.entries.delete(path); this.entries.set(path, entry);
    return entry;
  }

  hitAnchor(path: CanvasBezierPathContract, matrix: Mat2d, point: Vec2, scale: number, radius = 12): boolean {
    const entry = this.entry(path, matrix, scale), b = entry.bounds;
    const inside = point.x >= b.x - radius && point.y >= b.y - radius && point.x <= b.x + b.width + radius && point.y <= b.y + b.height + radius;
    const hit = inside && entry.anchors.some((anchor) => Math.hypot(anchor.x - point.x, anchor.y - point.y) <= radius);
    this.trim(); return hit;
  }

  find(path: CanvasBezierPathContract, matrix: Mat2d, point: Vec2, scale: number, radius = 10) {
    const entry = this.entry(path, matrix, scale), steps = entry.steps;
    const b = entry.bounds;
    if (point.x < b.x - radius || point.y < b.y - radius || point.x > b.x + b.width + radius || point.y > b.y + b.height + radius) {
      this.trim(); return null;
    }
    if (entry.samples.length === 0) {
      // Affine projection commutes with Bezier evaluation; sample once, reuse on every hover.
      entry.samples = approximateBezierPath(path.points, path.isClosed, steps).map((p) => applyToPoint(matrix, p));
      this.retained += entry.samples.length;
    }
    let best: { segmentIndex: number; t: number; point: Vec2; distance: number } | null = null;
    for (let i = 1; i < entry.samples.length; i++) {
      const a = entry.samples[i - 1]!, b = entry.samples[i]!;
      const dx = b.x - a.x, dy = b.y - a.y, length2 = dx * dx + dy * dy;
      const t = length2 === 0 ? 0 : Math.max(0, Math.min(1, ((point.x - a.x) * dx + (point.y - a.y) * dy) / length2));
      const nearest = { x: a.x + dx * t, y: a.y + dy * t };
      const distance = Math.hypot(point.x - nearest.x, point.y - nearest.y);
      if (distance <= radius && (!best || distance < best.distance)) {
        best = { segmentIndex: Math.floor((i - 1) / steps), t: ((i - 1) % steps + t) / steps, point: nearest, distance };
      }
    }
    this.trim();
    return best;
  }

  private trim(): void {
    while (this.entries.size > 64 || this.retained > MAX_CACHED_SAMPLES) {
      const oldest = this.entries.keys().next().value!;
      const entry = this.entries.get(oldest)!;
      this.retained -= entry.samples.length + entry.anchors.length;
      this.entries.delete(oldest);
    }
  }
}
