/**
 * Generation's preview-only public surface, kept separate from `graph.ts` so
 * consumers behind a lazy chunk boundary (the graph preview dialog) never
 * pull the preview compiler into `graph.ts`'s eager importers — mirrors
 * `@features/workflow/preview`.
 */
export { compileGeneratePreviewGraph, stabilizeBackendGraphIds } from './core/previewGraph';
export type { GeneratePreviewInput, GeneratePreviewResult } from './core/previewGraph';
export { getGenerateNodeProvenance } from './core/graphProvenance';
export type { GenerateProvenanceEntry } from './core/graphProvenance';
