/** Keep ordinary graph compilation eager and canvasGraph lazy. */
export {
  addLoraCollectionLoader,
  addTransformerLoraCollectionLoader,
  compileGenerateGraph,
  resolveGenerateSeed,
} from './core/graph';
export {
  addEdge,
  addNode,
  createId,
  getActiveCompatibleLoras,
  toGraphContract,
  toModelIdentifier,
} from './core/graphBuilder';
export { detectCanvasMode } from './core/canvas/canvasMode';
export {
  type ControlAdapterKind,
  type ControlValidationReason,
  getControlValidationReason,
  isControlKindSupportedForBase,
} from './core/canvas/controlValidation';
export { getControlLayerRejectionReason, getControlValidationReasonMessage } from './core/canvas/addControlLayers';
export {
  getRegionalGuidanceRejectionReason,
  getRegionalGuidanceSupport,
  isRegionalGuidanceSupportedForBase,
  type RegionalGuidanceInput,
  type RegionalReferenceImageInput,
} from './core/canvas/addRegionalGuidance';
export type { ControlLayerGraphInput } from './core/canvas/addControlLayers';
