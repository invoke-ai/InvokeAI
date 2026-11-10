import type { Vec2 } from '../types';

/** v6 taper profile: smoothstep pressure over up to half the path at either end. */
export const buildTaperedTracePoints = (coordinates: Vec2[], strokeWidth: number, taper = 100): [number, number, number][] => {
  const distances = coordinates.map(() => 0);
  for (let i = 1; i < coordinates.length; i++) {
    distances[i] = distances[i - 1]! + Math.hypot(coordinates[i]!.x - coordinates[i - 1]!.x, coordinates[i]!.y - coordinates[i - 1]!.y);
  }
  const total = distances.at(-1) ?? 0;
  const length = Math.min(total / 2, Math.max(0, strokeWidth * 6 * taper / 100));
  return coordinates.map((point, i) => {
    const t = length === 0 ? 1 : Math.min(1, Math.min(distances[i]!, total - distances[i]!) / length);
    return [point.x, point.y, t * t * (3 - 2 * t)];
  });
};
