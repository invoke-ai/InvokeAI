/**
 * Generation's eagerly shared graph surface: the generate compiler, Canvas
 * validation policies, and graph-builder primitives used by other invocation
 * sources. Canvas compilation lives in the lazy `canvasGraph` interface.
 * Curated, caller-driven export list — add a symbol only when a consumer needs it.
 */
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
  isRegionalGuidanceSupportedForBase,
  type RegionalGuidanceInput,
  type RegionalReferenceImageInput,
} from './core/canvas/addRegionalGuidance';
export type { ControlLayerGraphInput } from './core/canvas/addControlLayers';
