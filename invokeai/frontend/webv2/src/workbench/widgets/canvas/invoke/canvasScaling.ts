/** Persist processing scale in canvas values for compilation; none preserves bbox size snapped to model grid. */

import type { CanvasScaleMethod, CanvasScalingSettings } from '@features/generation/contracts';

export const CANVAS_SCALING_KEYS = {
  height: 'scaledHeight',
  method: 'scaleMethod',
  width: 'scaledWidth',
} as const;

export const CANVAS_SCALE_METHODS: readonly CanvasScaleMethod[] = ['none', 'auto', 'manual'];

/** No scaling: the bbox is processed as-is (snapped to the model grid). */
export const DEFAULT_CANVAS_SCALING: CanvasScalingSettings = { height: null, method: 'none', width: null };

const isScaleMethod = (value: unknown): value is CanvasScaleMethod =>
  CANVAS_SCALE_METHODS.includes(value as CanvasScaleMethod);

const readSide = (value: unknown): number | null =>
  typeof value === 'number' && Number.isFinite(value) && value > 0 ? Math.round(value) : null;

export const readCanvasScaling = (values: Record<string, unknown> | undefined): CanvasScalingSettings => {
  const method = values?.[CANVAS_SCALING_KEYS.method];
  return {
    height: readSide(values?.[CANVAS_SCALING_KEYS.height]),
    method: isScaleMethod(method) ? method : 'none',
    width: readSide(values?.[CANVAS_SCALING_KEYS.width]),
  };
};
