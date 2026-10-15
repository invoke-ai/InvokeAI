/**
 * Sync only for Canvas sources. Bbox edits retain the exact footprint and derive model-grid processing dimensions;
 * Generate edits resize the bbox at its top-left. Position-only changes and submit-only scaling do not
 * participate. Capability changes retrigger reconciliation; update the expected snapshot before dispatch and guard
 * re-entry.
 */

import type { AspectRatioId } from '@features/generation/contracts';
import type { CanvasDocumentContractV3 } from '@workbench/canvas-engine/api';
import type { WorkbenchState } from '@workbench/projectContracts';

import { subscribeArchitectureCapabilities } from '@features/generation/runtime';
import { clampDimension, deriveAspectRatioId } from '@features/generation/settings';

import type { WorkbenchCommands } from './workbenchStore';

import { resolveModelGrid } from './widgets/canvas/bboxGrid';
import { getProjectWidgetValues } from './widgetState';

type Bbox = CanvasDocumentContractV3['bbox'];

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
   * Bbox authority overrides locked aspect presets; derive both aspect values from the exact bbox while snapping
   * processing dimensions.
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
 * The changed side wins; bbox wins on first run or simultaneous changes. Write model-grid dimensions without
 * changing its exact footprint, and skip already-consistent state.
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

// Use the model variant because its grid can differ and reconciled dimensions are persisted.
const readModelVariant = (values: Record<string, unknown>): string | null => {
  const model = values.model;
  return model && typeof model === 'object' && typeof (model as { variant?: unknown }).variant === 'string'
    ? (model as { variant: string }).variant
    : null;
};

export const createCanvasDimsSync = (store: CanvasDimsSyncStore): CanvasDimsSync => {
  let prev: CanvasDimsSnapshot | null = null;
  let lastProjectId: string | null = null;
  let isSyncing = false;

  const handleChange = (): void => {
    // Dispatch synchronously re-enters this listener; skip the nested pass to bound writes.
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
    const grid = resolveModelGrid(readModelBase(generateValues), readModelVariant(generateValues));

    // Wait for backend capabilities before persisting dimensions; a fallback grid may be invalid for this
    // architecture.
    if (grid === null) {
      prev = null;
      return;
    }

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

  const unsubscribeStore = store.subscribe(handleChange);
  // Capability arrival may not change workbench state, so subscribe directly to retry the gated reconciliation.
  const unsubscribeCapabilities = subscribeArchitectureCapabilities(handleChange);

  // Seed from the current state so an already-canvas project reconciles on mount.
  handleChange();

  return {
    dispose: () => {
      unsubscribeStore();
      unsubscribeCapabilities();
    },
  };
};
