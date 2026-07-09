/**
 * The non-destructive control-filter preview orchestration (plan §1.4).
 *
 * A control layer can preview a filter (canny, depth, pose, …) over its
 * composited source WITHOUT mutating the document: this module composites the one
 * control layer, uploads it, builds the single-node filter graph, and runs it on
 * the utility queue (outside any project queue) to get a filtered image name.
 * The caller then shows it via the engine's `setFilterPreview` (a transient render
 * override) — "Apply" swaps the layer source (an undoable reducer edit), "Cancel"
 * clears the preview.
 *
 * It is React-free and fully injectable: `flushPendingUploads`, `getDocument`,
 * the composite `executorDeps`, and `runFilterGraph` are all parameters, so the
 * whole flow runs in node tests against fakes. The pure pieces it composes —
 * `buildFilterGraph` and `runUtilityGraph` — are unit-tested separately.
 */

import type { ExecuteCompositePlanDeps } from '@workbench/canvas-engine/export/compositeForGeneration';
import type { CompositeEntry } from '@workbench/generation/canvas/types';
import type { BackendGraphContract, CanvasDocumentContractV2 } from '@workbench/types';

import { executeControlComposite } from '@workbench/canvas-engine/export/compositeForGeneration';
import { planControlComposites } from '@workbench/generation/canvas/compositePlan';
import { buildFilterGraph } from '@workbench/generation/canvas/filterGraphs';

/**
 * The filtered preview image: its name, its dimensions (== the LAYER's content
 * rect, not the doc bbox), and `origin` — the layer-local top-left the filtered
 * pixels occupy. Apply swaps in an image source of exactly these dimensions
 * positioned at `origin`, so preview and apply are pixel-identical.
 */
export interface ControlFilterPreviewResult {
  imageName: string;
  width: number;
  height: number;
  /** Layer-local origin of the filtered content (the content rect's top-left). */
  origin: { x: number; y: number };
}

/** Injected dependencies for {@link runControlFilterPreview} (all side effects are seams). */
export interface RunControlFilterPreviewDeps {
  /** Persist any pending paint bitmaps before compositing from their pixels. */
  flushPendingUploads: () => Promise<void>;
  /** The live mirrored document (`document.bbox` frames the composite). */
  getDocument: () => CanvasDocumentContractV2 | null;
  /** Engine-backed composite executor deps (backend + rasterize + upload + dedupe). */
  executorDeps: ExecuteCompositePlanDeps;
  /** Runs a filter graph on the utility queue and resolves its output image name. */
  runFilterGraph: (graph: BackendGraphContract, outputNodeId: string, signal?: AbortSignal) => Promise<string>;
}

/** Thrown when the target control layer has no compositable content to filter. */
export class ControlFilterPreviewError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ControlFilterPreviewError';
  }
}

const throwIfAborted = (signal: AbortSignal | undefined): void => {
  if (signal?.aborted) {
    throw new DOMException('The control-filter preview was aborted.', 'AbortError');
  }
};

/**
 * Composites the one control layer `layerId`, uploads it, and runs `filterType`
 * over it on the utility queue, resolving with the filtered image name + dims.
 * Rejects with a {@link ControlFilterPreviewError} when the layer has no content,
 * or an `AbortError` when the `signal` fires. Never touches the document — the
 * caller applies or discards the result.
 */
export const runControlFilterPreview = async (params: {
  layerId: string;
  filterType: string;
  settings?: Record<string, unknown>;
  deps: RunControlFilterPreviewDeps;
  signal?: AbortSignal;
}): Promise<ControlFilterPreviewResult> => {
  const { deps, filterType, layerId, settings, signal } = params;

  await deps.flushPendingUploads();
  throwIfAborted(signal);

  const document = deps.getDocument();
  if (!document) {
    throw new ControlFilterPreviewError('The canvas has no active document to filter.');
  }

  // Only content-bearing control layers appear here (empty ones are excluded).
  const planned = planControlComposites(document, document.bbox).find(
    (composite) => composite.layerId === layerId
  )?.entry;
  if (!planned) {
    throw new ControlFilterPreviewError('This control layer has no content to filter.');
  }

  const ref = planned.layers[0];
  if (!ref) {
    throw new ControlFilterPreviewError('This control layer has no content to filter.');
  }

  // Scope the composite to the LAYER's own content rect, NOT the doc bbox. The
  // compositor draws the filter preview over the layer's content footprint (its
  // cache rect), so a bbox-sized composite would be stretched/offset against the
  // preview overlay and Apply would crop content lying outside the bbox. We
  // composite the layer's content-sized region with an IDENTITY transform (the
  // layer transform is applied by the compositor/Apply separately — baking it
  // here too would double it), then Apply swaps in an image of exactly these
  // dimensions at `origin`.
  const origin = { x: ref.contentOffset.x, y: ref.contentOffset.y };
  const contentRect = {
    height: ref.contentSize.height,
    width: ref.contentSize.width,
    x: origin.x,
    y: origin.y,
  };
  const entry: CompositeEntry = {
    bbox: contentRect,
    key: `control-filter|${layerId}|${contentRect.x},${contentRect.y},${contentRect.width},${contentRect.height}|${ref.sourceRef}`,
    kind: 'control-layer',
    layerId,
    layers: [{ ...ref, transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 } }],
  };

  const composite = await executeControlComposite(entry, deps.executorDeps);
  throwIfAborted(signal);

  const { graph, outputNodeId } = buildFilterGraph(filterType, composite.imageName, settings);
  const imageName = await deps.runFilterGraph(graph, outputNodeId, signal);
  throwIfAborted(signal);

  return { height: composite.height, imageName, origin, width: composite.width };
};
