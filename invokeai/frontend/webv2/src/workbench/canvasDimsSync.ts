/**
 * Two-way sync between a project's canvas generation frame (`canvas.document.bbox`)
 * and its model-valid generate-widget processing dimensions (`width` / `height` /
 * `aspectRatioId`).
 *
 * Legacy-compatible geometry: the generation frame remains the exact final
 * canvas footprint, including off-grid sizes. Resizing the bbox (tool gesture,
 * BboxOptions, undo/redo) drives width/height snapped to the selected model's
 * hard processing grid; the canvas graph resizes inputs to that processing size
 * and the result back to the bbox. Unlike legacy's optional "Scale Before
 * Processing" policy, this does not silently upscale small bboxes to an optimal
 * pixel area. Editing the generate dimensions (or picking an aspect preset)
 * still resizes the bbox in place (top-left anchored). Position-only bbox moves
 * never touch the dimensions.
 *
 * The module is two parts:
 * - {@link reconcileCanvasDims}: a pure, unit-tested reconcile that decides which
 *   side (if any) to write, given the current bbox, the current committed
 *   generate dims, the snapping grid, and the last-synced snapshot.
 * - {@link createCanvasDimsSync}: a thin `store.subscribe` wiring that feeds the
 *   reconcile from workbench state and dispatches the resulting action.
 *
 * Loop safety: the reconcile short-circuits to `none` whenever the bbox and dims
 * already agree, so applying either direction is a fixed point. The wiring also
 * updates its last-synced snapshot *before* dispatching and hard-guards against
 * re-entrant notifications, keeping the dispatch count per external change
 * bounded (at most one echo, which is itself a no-op).
 *
 * Only active while `project.invocation.sourceId === 'canvas'`; for every other
 * source the sync is inert and generate-dimension editing behaves exactly as it
 * does today. Zero React.
 */

import type { AspectRatioId } from '@features/generation/contracts';
import type { CanvasDocumentContractV2 } from '@workbench/canvas-engine/api';
import type { WorkbenchState } from '@workbench/projectContracts';

import { clampDimension, deriveAspectRatioId } from '@features/generation/settings';

import type { WorkbenchCommands } from './workbenchStore';

import { gridSizeForModelBase } from './widgets/canvas/bboxGrid';
import { getProjectWidgetValues } from './widgetState';

type Bbox = CanvasDocumentContractV2['bbox'];

/** The last-synced width/height on both sides, used to detect which side changed. */
export interface CanvasDimsSnapshot {
  bboxWidth: number;
  bboxHeight: number;
  dimsWidth: number;
  dimsHeight: number;
  grid: number;
}

export interface CanvasDimsReconcileInput {
  /** The current generation frame, or `null` when the sync should stay inert (no canvas mode). */
  bbox: Bbox | null;
  /** The current committed generate width/height. */
  dims: { width: number; height: number };
  /** The bbox/generate snapping grid (model-derived; identical on both sides). */
  grid: number;
  /** The last snapshot this sync wrote/observed, or `null` on first run / after a reset. */
  prev: CanvasDimsSnapshot | null;
}

export type CanvasDimsReconcileResult =
  | { kind: 'none' }
  /**
   * Write a model-grid processing size onto the generate dims (bbox wins),
   * retaining the exact bbox ratio in the aspect controls. Both aspect values
   * are re-derived from the bbox unconditionally (even when the form's ratio is
   * locked to a preset) so a locked preset does not veto the bbox, which remains
   * authoritative.
   */
  | { kind: 'patch-dims'; width: number; height: number; aspectRatioId: AspectRatioId; aspectRatioValue: number }
  /** Resize the bbox to the (grid-snapped) generate dims, keeping its top-left position. */
  | { kind: 'set-bbox'; bbox: Bbox };

const getDimsPatch = (width: number, height: number, aspectWidth = width, aspectHeight = height) => ({
  aspectRatioId: deriveAspectRatioId(aspectWidth, aspectHeight),
  aspectRatioValue: aspectHeight > 0 ? aspectWidth / aspectHeight : 1,
  height,
  width,
});

const createSnapshot = (
  bbox: Pick<Bbox, 'width' | 'height'>,
  dims: { width: number; height: number },
  grid: number
): CanvasDimsSnapshot => ({
  bboxHeight: bbox.height,
  bboxWidth: bbox.width,
  dimsHeight: dims.height,
  dimsWidth: dims.width,
  grid,
});

/**
 * Decide which direction of the bbox <-> dims sync to apply.
 *
 * - No bbox (not in canvas mode) -> `none`.
 * - A grid-valid bbox and dims that already agree -> `none` (the primary loop guard).
 * - Otherwise the side that changed since `prev` wins; the bbox is authoritative
 *   when both (or neither, on first run) changed. Bbox -> dims writes the nearest
 *   model-grid processing size while preserving the exact bbox footprint and
 *   aspect. Dims -> bbox snaps to the grid and only emits when the snapped size
 *   actually differs from the live bbox.
 */
export const reconcileCanvasDims = ({
  bbox,
  dims,
  grid,
  prev,
}: CanvasDimsReconcileInput): CanvasDimsReconcileResult => {
  if (!bbox) {
    return { kind: 'none' };
  }

  const bboxIsOnGrid = bbox.width % grid === 0 && bbox.height % grid === 0;
  const gridChanged = prev && prev.grid !== grid;

  if (bboxIsOnGrid && !gridChanged && bbox.width === dims.width && bbox.height === dims.height) {
    return { kind: 'none' };
  }

  const bboxChanged = !prev || gridChanged || prev.bboxWidth !== bbox.width || prev.bboxHeight !== bbox.height;

  if (bboxChanged) {
    const width = clampDimension(bbox.width, grid);
    const height = clampDimension(bbox.height, grid);

    return { ...getDimsPatch(width, height, bbox.width, bbox.height), kind: 'patch-dims' };
  }

  const dimsChanged = prev!.dimsWidth !== dims.width || prev!.dimsHeight !== dims.height;

  if (dimsChanged) {
    const width = clampDimension(dims.width, grid);
    const height = clampDimension(dims.height, grid);

    if (width === bbox.width && height === bbox.height) {
      return { kind: 'none' };
    }

    return { bbox: { height, width, x: bbox.x, y: bbox.y }, kind: 'set-bbox' };
  }

  return { kind: 'none' };
};

/** The minimal workbench store surface the sync depends on. */
export interface CanvasDimsSyncStore {
  commands: {
    canvas: Pick<WorkbenchCommands['canvas'], 'apply'>;
    generation: Pick<WorkbenchCommands['generation'], 'patchSettings'>;
  };
  getState(): WorkbenchState;
  subscribe(listener: () => void): () => void;
}

export interface CanvasDimsSync {
  dispose(): void;
}

const readFiniteDimension = (values: Record<string, unknown>, key: 'width' | 'height'): number | null => {
  const raw = values[key];
  return Number.isFinite(raw as number) && (raw as number) > 0 ? (raw as number) : null;
};

const readModelBase = (values: Record<string, unknown>): string | null => {
  const model = values.model;
  return model && typeof model === 'object' && typeof (model as { base?: unknown }).base === 'string'
    ? (model as { base: string }).base
    : null;
};

/**
 * Wire the bbox <-> generate-dims reconcile onto a workbench store. Subscribes
 * immediately; dispatches `patchGenerateSettings` / `setCanvasBbox` as the
 * reconcile directs. Returns a handle whose `dispose` removes the subscription.
 */
export const createCanvasDimsSync = (store: CanvasDimsSyncStore): CanvasDimsSync => {
  let prev: CanvasDimsSnapshot | null = null;
  let lastProjectId: string | null = null;
  let isSyncing = false;

  const handleChange = (): void => {
    // A dispatch below re-enters this listener synchronously; the snapshot is
    // already updated to the post-dispatch expectation, so the nested pass would
    // be a no-op — skip it to keep the dispatch count strictly bounded.
    if (isSyncing) {
      return;
    }

    const state = store.getState();
    const project = state.projects.find((candidate) => candidate.id === state.activeProjectId);

    if (!project) {
      prev = null;
      lastProjectId = null;
      return;
    }

    if (project.id !== lastProjectId) {
      lastProjectId = project.id;
      prev = null;
    }

    // Inert unless the project is invoking into the canvas: for every other
    // source the generate dimensions behave exactly as they do today.
    if (project.invocation.sourceId !== 'canvas') {
      prev = null;
      return;
    }

    const generateValues = getProjectWidgetValues(project, 'generate');
    const width = readFiniteDimension(generateValues, 'width');
    const height = readFiniteDimension(generateValues, 'height');

    if (width === null || height === null) {
      prev = null;
      return;
    }

    const bbox = project.canvas.document.bbox;
    const grid = gridSizeForModelBase(readModelBase(generateValues));
    const result = reconcileCanvasDims({ bbox, dims: { height, width }, grid, prev });
    const projectId = project.id;

    switch (result.kind) {
      case 'none': {
        prev = createSnapshot(bbox, { height, width }, grid);
        return;
      }
      case 'patch-dims': {
        const { kind: _, ...patch } = result;
        prev = createSnapshot(bbox, result, grid);
        isSyncing = true;
        try {
          store.commands.generation.patchSettings(patch, projectId, 'system');
        } finally {
          isSyncing = false;
        }
        return;
      }
      case 'set-bbox': {
        const nextBbox = result.bbox;
        prev = createSnapshot(nextBbox, nextBbox, grid);
        isSyncing = true;
        try {
          store.commands.canvas.apply(projectId, { bbox: nextBbox, type: 'setCanvasBbox' }, 'system');
          if (width !== nextBbox.width || height !== nextBbox.height) {
            store.commands.generation.patchSettings(getDimsPatch(nextBbox.width, nextBbox.height), projectId, 'system');
          }
        } finally {
          isSyncing = false;
        }
        return;
      }
    }
  };

  const unsubscribe = store.subscribe(handleChange);

  // Seed from the current state so an already-canvas project reconciles on mount.
  handleChange();

  return { dispose: unsubscribe };
};
