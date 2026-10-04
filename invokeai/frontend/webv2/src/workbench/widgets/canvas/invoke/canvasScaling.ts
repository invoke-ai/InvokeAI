/** Persist processing scale in canvas values for compilation; none preserves bbox size snapped to model grid. */

import type { CanvasScaleMethod, CanvasScalingSettings } from '@features/generation/contracts';

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
