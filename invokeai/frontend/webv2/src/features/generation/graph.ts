/**
 * Generation's graph-compilation surface: generate/canvas graph compilers and
 * the graph-builder primitives shared with other invocation sources.
 * Curated, caller-driven export list — add a symbol only when a consumer needs it.
 */
export { addLoraCollectionLoader, compileGenerateGraph, resolveGenerateSeed } from './core/graph';
export { addEdge, addNode, getActiveCompatibleLoras, toGraphContract, toModelIdentifier } from './core/graphBuilder';
export { compileCanvasGraph } from './core/canvas/compileCanvasGraph';
export { detectCanvasMode } from './core/canvas/canvasMode';
export {
  type ControlAdapterKind,
  getControlValidationReason,
  isControlKindSupportedForBase,
} from './core/canvas/controlValidation';
export {
  getRegionalGuidanceRejectionReason,
  isRegionalGuidanceSupportedForBase,
  type RegionalGuidanceInput,
  type RegionalReferenceImageInput,
} from './core/canvas/addRegionalGuidance';
export type { ControlLayerGraphInput } from './core/canvas/addControlLayers';
export type { CanvasCompileMode } from './core/canvas/types';
