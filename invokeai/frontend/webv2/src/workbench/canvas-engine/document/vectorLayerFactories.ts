import type { CanvasVectorLayerContract } from '@workbench/canvas-engine/contracts';

import { DEFAULT_VECTOR_STROKE_COLOR, DEFAULT_VECTOR_STROKE_WIDTH } from '@workbench/canvas-engine/vectorDefaults';

import { nextNumberedName } from './layerFactories';

// Vector creation stays separate from the raster/control factories used by non-Canvas image actions.
export const nextVectorLayerName = (names: readonly string[]): string => nextNumberedName('Vector Layer', names);

export const createVectorLayer = (input: { id: string; name: string }): CanvasVectorLayerContract => ({
  blendMode: 'normal',
  id: input.id,
  isEnabled: true,
  isLocked: false,
  name: input.name,
  opacity: 1,
  paths: [],
  strokeColor: DEFAULT_VECTOR_STROKE_COLOR,
  strokeWidth: DEFAULT_VECTOR_STROKE_WIDTH,
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'vector',
});
