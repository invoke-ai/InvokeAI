/**
 * Generation's graph-compilation surface: generate/canvas graph compilers and
 * the graph-builder primitives shared with other invocation sources.
 * Curated, caller-driven export list — add a symbol only when a consumer needs it.
 */
export { addLoraCollectionLoader, compileGenerateGraph, resolveGenerateSeed } from './core/graph';
export { compileGeneratePreviewGraph, stabilizeBackendGraphIds } from './core/previewGraph';
export type { GeneratePreviewInput, GeneratePreviewResult } from './core/previewGraph';
export { getGenerateNodeProvenance } from './core/graphProvenance';
export type { GenerateProvenanceEntry } from './core/graphProvenance';
export { addEdge, addNode, getActiveCompatibleLoras, toGraphContract, toModelIdentifier } from './core/graphBuilder';
export { compileCanvasGraph } from './core/canvas/compileCanvasGraph';
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
export type { CanvasCompileMode } from './core/canvas/types';
