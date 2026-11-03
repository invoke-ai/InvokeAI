/**
 * Sole caller interface for the complete Canvas module.
 *
 * The rendering core depends only on `capabilities.ts`; this facade may compose
 * core capabilities with Canvas-owned application workflows without reversing
 * the core dependency direction.
 */
export * from './capabilities';
export { CONTROL_TYPES } from './contracts';
export {
  controlDetectionInputKey,
  getControlDetectionInput,
  isControlMapCurrent,
  needsControlDetection,
  restoreControlOriginal,
} from './document/controlDerivation';
export { createControlLayer, type CreateControlLayerInput, nextControlLayerName } from './document/layerFactories';
