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
  getLoadedLoras,
  toGraphContract,
  toModelIdentifier,
} from './core/graphBuilder';
export { detectCanvasMode } from './core/canvas/canvasMode';
export {
  areControlAdapterValuesValid,
  CONTROL_ADAPTER_KINDS,
  CONTROL_VALIDATION_REASONS,
  type ControlAdapterKind,
  type ControlValidationReason,
  createControlValidationSequence,
  getControlModelUnusableReason,
  getControlValidationReason,
  getSuggestedControlKind,
  isControlKindSupportedForBase,
  isControlModelUsableForKind,
} from './core/canvas/controlValidation';
export {
  getRegionalGuidanceRejectionReason,
  getRegionalGuidanceSupport,
  isRegionalGuidanceSupportedForBase,
  type RegionalGuidanceInput,
  type RegionalReferenceImageInput,
} from './core/canvas/addRegionalGuidance';
export type { ControlLayerGraphInput } from './core/canvas/addControlLayers';
