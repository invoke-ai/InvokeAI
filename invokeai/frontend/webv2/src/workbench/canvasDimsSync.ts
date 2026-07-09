/**
 * Two-way sync between a project's canvas generation frame (`canvas.document.bbox`)
 * and its generate-widget dimensions (`width` / `height` / `aspectRatioId`).
 *
 * Legacy parity: while a project is invoking into the canvas, the generation
 * frame IS the generation size. Resizing the bbox (tool gesture, BboxOptions,
 * undo/redo) drives the generate width/height; editing the generate dimensions
 * (or picking an aspect preset) resizes the bbox in place (top-left anchored).
 * Position-only bbox moves never touch the dimensions.
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

import type { AspectRatioId } from './generation/types';
import type { CanvasDocumentContractV2, WorkbenchState } from './types';
import type { WorkbenchAction } from './workbenchState';

import { deriveAspectRatioId } from './generation/settings';
import { gridSizeForModelBase } from './widgets/canvas/bboxGrid';
import { getProjectWidgetValues } from './widgetState';

type Bbox = CanvasDocumentContractV2['bbox'];

/** The last-synced width/height on both sides, used to detect which side changed. */
export interface CanvasDimsSnapshot {
  bboxWidth: number;
  bboxHeight: number;
  dimsWidth: number;
  dimsHeight: number;
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
   * Write the bbox size onto the generate dims (bbox wins), re-deriving the
   * aspect id *and* the numeric aspect ratio. Both are re-derived from the bbox
   * unconditionally (even when the form's ratio is locked to a preset) so the
   * id, the numeric ratio, and the dims stay mutually consistent after the
   * patch — a locked preset does not veto the bbox, which remains authoritative.
   */
  | { kind: 'patch-dims'; width: number; height: number; aspectRatioId: AspectRatioId; aspectRatioValue: number }
  /** Resize the bbox to the (grid-snapped) generate dims, keeping its top-left position. */
  | { kind: 'set-bbox'; bbox: Bbox };

const snapToGrid = (value: number, grid: number): number => {
  const step = grid > 0 ? grid : 1;
  return Math.max(step, Math.round(value / step) * step);
};

/**
 * Decide which direction of the bbox <-> dims sync to apply.
 *
 * - No bbox (not in canvas mode) -> `none`.
 * - Bbox and dims already agree -> `none` (the primary loop guard).
 * - Otherwise the side that changed since `prev` wins; the bbox is authoritative
 *   when both (or neither, on first run) changed, matching "the frame is the
 *   generation size". Bbox -> dims writes the exact bbox size and re-derives the
 *   aspect id. Dims -> bbox snaps to the grid and only emits when the snapped
 *   size actually differs from the live bbox.
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

  if (bbox.width === dims.width && bbox.height === dims.height) {
    return { kind: 'none' };
  }

  const bboxChanged = !prev || prev.bboxWidth !== bbox.width || prev.bboxHeight !== bbox.height;

  if (bboxChanged) {
    return {
      aspectRatioId: deriveAspectRatioId(bbox.width, bbox.height),
      aspectRatioValue: bbox.height > 0 ? bbox.width / bbox.height : 1,
      height: bbox.height,
      kind: 'patch-dims',
      width: bbox.width,
    };
  }

  const dimsChanged = !prev || prev.dimsWidth !== dims.width || prev.dimsHeight !== dims.height;

  if (dimsChanged) {
    const width = snapToGrid(dims.width, grid);
    const height = snapToGrid(dims.height, grid);

    if (width === bbox.width && height === bbox.height) {
      return { kind: 'none' };
    }

    return { bbox: { height, width, x: bbox.x, y: bbox.y }, kind: 'set-bbox' };
  }

  return { kind: 'none' };
};

/** The minimal workbench store surface the sync depends on. */
export interface CanvasDimsSyncStore {
  getState(): WorkbenchState;
  subscribe(listener: () => void): () => void;
  dispatch(action: WorkbenchAction): void;
}

export interface CanvasDimsSync {
  dispose(): void;
}

const readFiniteDimension = (values: Record<string, unknown>, key: 'width' | 'height'): number | null => {
  const raw = values[key];
  return typeof raw === 'number' && Number.isFinite(raw) && raw > 0 ? raw : null;
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

    switch (result.kind) {
      case 'none': {
        prev = { bboxHeight: bbox.height, bboxWidth: bbox.width, dimsHeight: height, dimsWidth: width };
        return;
      }
      case 'patch-dims': {
        prev = {
          bboxHeight: bbox.height,
          bboxWidth: bbox.width,
          dimsHeight: result.height,
          dimsWidth: result.width,
        };
        isSyncing = true;
        try {
          store.dispatch({
            projectId: project.id,
            type: 'patchGenerateSettings',
            values: {
              aspectRatioId: result.aspectRatioId,
              aspectRatioValue: result.aspectRatioValue,
              height: result.height,
              width: result.width,
            },
          });
        } finally {
          isSyncing = false;
        }
        return;
      }
      case 'set-bbox': {
        prev = {
          bboxHeight: result.bbox.height,
          bboxWidth: result.bbox.width,
          dimsHeight: height,
          dimsWidth: width,
        };
        isSyncing = true;
        try {
          store.dispatch({ bbox: result.bbox, type: 'setCanvasBbox' });
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
