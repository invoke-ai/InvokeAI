import type { Vec2 } from '../types';

type Segment = { a: Vec2; b: Vec2; cuts: number[]; minX: number; maxX: number; minY: number; maxY: number };
type Vertex = { point: Vec2; outgoing: number[] };
type Edge = { from: number; to: number; reverse: number; visited: boolean };

const cross = (a: Vec2, b: Vec2) => a.x * b.y - a.y * b.x;
const subtract = (a: Vec2, b: Vec2): Vec2 => ({ x: a.x - b.x, y: a.y - b.y });
const EPSILON = 1e-7;

/** Experimental planar-face extraction. Input is sampled in layer coordinates, independent of zoom.
 * Only actual crossings connect paths; gaps and collinear overlaps are not repaired.
 */
export const VECTOR_FILL_LIMITS = { segments: 24_000, comparisons: 2_000_000, edges: 120_000 } as const;
export class VectorFillLimitError extends Error {
  constructor() { super('Vector fill geometry exceeds the supported work budget'); this.name = 'VectorFillLimitError'; }
}
type Limits = { segments: number; comparisons: number; edges: number };

function* regionSteps(paths: readonly (readonly Vec2[])[], limits: Limits): Generator<void, Vec2[][]> {
  let work = 0;
  yield;
  const segments: Segment[] = [];
  for (const points of paths) {
    for (let i = 1; i < points.length; i++) {
      const a = points[i - 1]!, b = points[i]!;
      if (Math.hypot(b.x - a.x, b.y - a.y) <= EPSILON) { continue; }
      if (segments.length >= limits.segments) { throw new VectorFillLimitError(); }
      segments.push({ a, b, cuts: [0, 1], minX: Math.min(a.x, b.x), maxX: Math.max(a.x, b.x),
        minY: Math.min(a.y, b.y), maxY: Math.max(a.y, b.y) });
    }
  }
  segments.sort((a, b) => a.minX - b.minX);
  let comparisons = 0;
  for (let i = 0; i < segments.length; i++) {
    const a = segments[i]!, r = subtract(a.b, a.a);
    for (let j = i + 1; j < segments.length; j++) {
      if (++comparisons > limits.comparisons) { throw new VectorFillLimitError(); }
      if (++work % 8192 === 0) { yield; }
      const b = segments[j]!;
      if (b.minX > a.maxX + EPSILON) { break; }
      if (b.minY > a.maxY + EPSILON || b.maxY < a.minY - EPSILON) { continue; }
      const s = subtract(b.b, b.a), denominator = cross(r, s);
      // Relative parallel test; overlapping edges are deliberately outside this prototype's scope.
      if (Math.abs(denominator) <= 1e-12 * Math.hypot(r.x, r.y) * Math.hypot(s.x, s.y)) { continue; }
      const delta = subtract(b.a, a.a);
      const t = cross(delta, s) / denominator, u = cross(delta, r) / denominator;
      if (t < -EPSILON || t > 1 + EPSILON || u < -EPSILON || u > 1 + EPSILON) { continue; }
      a.cuts.push(Math.max(0, Math.min(1, t)));
      b.cuts.push(Math.max(0, Math.min(1, u)));
    }
  }

  const vertices: Vertex[] = [], edges: Edge[] = [];
  const vertexIds = new Map<string, number>(), connections = new Set<string>();
  const vertex = (point: Vec2): number => {
    // Numerical identity only, not a user-visible snapping tolerance.
    const key = `${Math.round(point.x / EPSILON)},${Math.round(point.y / EPSILON)}`;
    const existing = vertexIds.get(key);
    if (existing !== undefined) { return existing; }
    const id = vertices.length;
    vertices.push({ point, outgoing: [] }); vertexIds.set(key, id);
    return id;
  };
  for (const segment of segments) {
    const cuts = segment.cuts.sort((a, b) => a - b);
    const at = (t: number): Vec2 => ({ x: segment.a.x + (segment.b.x - segment.a.x) * t,
      y: segment.a.y + (segment.b.y - segment.a.y) * t });
    for (let i = 1; i < cuts.length; i++) {
      if (cuts[i]! - cuts[i - 1]! <= 1e-12) { continue; }
      const from = vertex(at(cuts[i - 1]!)), to = vertex(at(cuts[i]!));
      const key = `${Math.min(from, to)}:${Math.max(from, to)}`;
      if (from === to || connections.has(key)) { continue; }
      if (edges.length + 2 > limits.edges) { throw new VectorFillLimitError(); }
      if (++work % 8192 === 0) { yield; }
      connections.add(key);
      const id = edges.length;
      edges.push({ from, to, reverse: id + 1, visited: false }, { from: to, to: from, reverse: id, visited: false });
      vertices[from]!.outgoing.push(id); vertices[to]!.outgoing.push(id + 1);
    }
  }
  for (const vertex of vertices) {
    const angle = (id: number) => {
      const to = vertices[edges[id]!.to]!.point;
      return Math.atan2(to.y - vertex.point.y, to.x - vertex.point.x);
    };
    vertex.outgoing.sort((a, b) => angle(a) - angle(b));
  }
  const regions: Vec2[][] = [];
  for (let start = 0; start < edges.length; start++) {
    if (edges[start]!.visited) { continue; }
    const points: Vec2[] = [];
    let current = start;
    do {
      const edge = edges[current]!;
      if (edge.visited) { break; }
      if (++work % 8192 === 0) { yield; }
      edge.visited = true;
      points.push(vertices[edge.from]!.point);
      const outgoing = vertices[edge.to]!.outgoing;
      const reverseIndex = outgoing.indexOf(edge.reverse);
      current = outgoing[(reverseIndex + outgoing.length - 1) % outgoing.length]!;
    } while (current !== start);
    if (current !== start || points.length < 3) { continue; }
    // Bounded faces have positive signed area; the unbounded exterior has negative area.
    const area = points.reduce((sum, point, i) => sum + cross(point, points[(i + 1) % points.length]!), 0) / 2;
    if (area > EPSILON) { regions.push(points); }
  }
  return regions;
}

/** Synchronous pure entry for small geometry and deterministic callers. */
export const findVectorFillRegions = (paths: readonly (readonly Vec2[])[]): Vec2[][] => {
  const steps = regionSteps(paths, VECTOR_FILL_LIMITS);
  let step = steps.next();
  while (!step.done) { step = steps.next(); }
  return step.value;
};

/** Yield bounded chunks to the browser so edits and Escape/Cancel can interrupt region discovery. */
export const findVectorFillRegionsAsync = async (
  paths: readonly (readonly Vec2[])[],
  options: { signal?: AbortSignal; yieldControl?: () => Promise<void>; limits?: Limits } = {}
): Promise<Vec2[][]> => {
  const steps = regionSteps(paths, options.limits ?? VECTOR_FILL_LIMITS);
  const yieldControl = options.yieldControl ?? (() => new Promise<void>((resolve) => setTimeout(resolve, 0)));
  try {
    while (true) {
      options.signal?.throwIfAborted();
      const step = steps.next();
      if (step.done) { return step.value; }
      await yieldControl();
    }
  } finally { steps.return([]); }
};
