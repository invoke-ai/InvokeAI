export const CURVE_SIZE = 180;
export const CURVE_PADDING = 6;

const CURVE_DRAW_SIZE = CURVE_SIZE - CURVE_PADDING * 2;
const clamp255 = (value: number): number => Math.max(0, Math.min(255, value));

export const getCurveGridCoordinates = (): number[] =>
  Array.from({ length: 5 }, (_, index) => CURVE_PADDING + (index / 4) * CURVE_DRAW_SIZE);

export const resolveCurveDragEnd = <TRestore, TCommit>(
  cancelled: boolean,
  before: TRestore,
  current: TCommit
): { commit: TCommit | null; restore: TRestore | null } =>
  cancelled ? { commit: null, restore: before } : { commit: current, restore: null };

export const curvePointToSvg = (x: number, y: number): { cx: number; cy: number } => ({
  cx: CURVE_PADDING + (clamp255(x) / 255) * CURVE_DRAW_SIZE,
  cy: CURVE_SIZE - CURVE_PADDING - (clamp255(y) / 255) * CURVE_DRAW_SIZE,
});

export const curvePointFromSvg = (px: number, py: number): [number, number] => {
  const x = Math.round(((px - CURVE_PADDING) / CURVE_DRAW_SIZE) * 255);
  const y = Math.round(((CURVE_SIZE - CURVE_PADDING - py) / CURVE_DRAW_SIZE) * 255);
  return [clamp255(x), clamp255(y)];
};
