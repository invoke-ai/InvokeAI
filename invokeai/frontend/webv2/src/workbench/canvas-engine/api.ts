/**
 * Sole caller interface for the complete Canvas module.
 *
 * The rendering core depends only on `capabilities.ts`; this facade may compose core capabilities with Canvas-owned
 * application workflows without reversing the core dependency direction.
 */
export * from './capabilities';
export { CONTROL_TYPES } from './contracts';
export { DEFAULT_VECTOR_GUIDE_COLOR, DEFAULT_VECTOR_STROKE_COLOR, DEFAULT_VECTOR_STROKE_WIDTH } from './contracts';
export {
  controlDetectionInputKey,
  getControlDetectionInput,
  isControlMapCurrent,
  needsControlDetection,
  restoreControlOriginal,
} from './document/controlDerivation';
export {
  advanceMaskFillColors,
  createControlLayer,
  type CreateControlLayerInput,
  DEFAULT_INPAINT_MASK_FILL,
  DEFAULT_REGIONAL_GUIDANCE_FILL_COLOR,
  nextControlLayerName,
  nextMaskFillColor,
} from './document/layerFactories';
export { createVectorLayer, nextVectorLayerName } from './document/vectorLayerFactories';
export { shapeDrawsOnLayer } from './tools/shapeTarget';
