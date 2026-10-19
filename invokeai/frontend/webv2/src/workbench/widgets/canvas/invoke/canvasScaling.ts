/** Persist processing scale in canvas values for compilation; none preserves bbox size snapped to model grid. */

import type { AspectRatioId, CanvasScaleMethod, CanvasScalingSettings } from '@features/generation/contracts';

import { calculateNewSize, clampDimension, isAspectRatioLocked } from '@features/generation/settings';

export const CANVAS_SCALING_KEYS = {
  height: 'scaledHeight',
  method: 'scaleMethod',
  width: 'scaledWidth',
} as const;

export const CANVAS_SCALE_METHODS: readonly CanvasScaleMethod[] = ['none', 'auto', 'manual'];

/** Auto: a small bbox grows to the model's optimal area, so canvases that never chose a method still process well. */
export const DEFAULT_CANVAS_SCALING: CanvasScalingSettings = { height: null, method: 'auto', width: null };

const isScaleMethod = (value: unknown): value is CanvasScaleMethod =>
  CANVAS_SCALE_METHODS.includes(value as CanvasScaleMethod);

const readSide = (value: unknown): number | null =>
  typeof value === 'number' && Number.isFinite(value) && value > 0 ? Math.round(value) : null;

export const readCanvasScaling = (values: Record<string, unknown> | undefined): CanvasScalingSettings => {
  const method = values?.[CANVAS_SCALING_KEYS.method];
  return {
    height: readSide(values?.[CANVAS_SCALING_KEYS.height]),
    method: isScaleMethod(method) ? method : DEFAULT_CANVAS_SCALING.method,
    width: readSide(values?.[CANVAS_SCALING_KEYS.width]),
  };
};

/** The ratio a manual render size keeps: the Size section's ratio while it is locked, otherwise null. */
export const readLockedAspectRatio = (generate: Record<string, unknown>): number | null => {
  const { aspectRatioId, aspectRatioIsLocked, aspectRatioValue } = generate;
  const isLocked = isAspectRatioLocked({
    aspectRatioId: typeof aspectRatioId === 'string' ? (aspectRatioId as AspectRatioId) : 'Free',
    aspectRatioIsLocked: aspectRatioIsLocked === true,
  });

  return isLocked && typeof aspectRatioValue === 'number' && Number.isFinite(aspectRatioValue) && aspectRatioValue > 0
    ? aspectRatioValue
    : null;
};

/**
 * Carry a manual render size to a new locked ratio at about the same area, on the model grid. Null when it already
 * holds the ratio, meaning one side is the other's grid-snapped counterpart (as the Size section derives a locked side).
 */
export const fitCanvasScalingToRatio = (
  size: { height: number; width: number },
  ratio: number,
  grid: number
): { height: number; width: number } | null =>
  clampDimension(size.width / ratio, grid) === size.height || clampDimension(size.height * ratio, grid) === size.width
    ? null
    : calculateNewSize(ratio, size.width * size.height, grid);
