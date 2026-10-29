/**
 * Sync only for Canvas sources: then the canvas frame and Generate's size are one frame-size model. Bbox edits keep
 * the exact footprint and derive model-grid processing dimensions; Generate edits resize the frame around its centre,
 * the anchor every size path uses. The aspect lock is Generate's own (`aspectRatioIsLocked`), and the Frame tool
 * follows it: a frame edit under the lock keeps the locked ratio, one that changes the ratio carries the lock to the
 * new ratio, and a free edit leaves the ratio free (it never reads as a preset the person did not pick).
 * A frame resize this sync makes goes through the canvas's undo history when a canvas engine is open, so undoing it
 * restores both sides; otherwise it is a `system` write. The Custom render size follows the locked ratio. Capability
 * changes retrigger reconciliation; update the expected snapshot before dispatch and guard re-entry.
 */

import type { AspectRatioId, CanvasScaleMethod, GenerateModelConfig, PidMode } from '@features/generation/contracts';
import type { CanvasDocumentContractV3 } from '@workbench/canvas-engine/api';
import type { WorkbenchState } from '@workbench/projectContracts';

import { subscribeArchitectureCapabilities } from '@features/generation/runtime';
import { clampDimension, deriveAspectRatioId, getGenerationDimensions } from '@features/generation/settings';

import type { WorkbenchCommands } from './workbenchStore';

import { resolveModelGrid } from './widgets/canvas/bboxGrid';
import {
  CANVAS_SCALING_KEYS,
  fitCanvasScalingToRatio,
  readCanvasScaling,
  readLockedAspectRatio,
} from './widgets/canvas/invoke/canvasScaling';
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

/** Generate's aspect fields: the one aspect lock the frame and the Size section share. */
export interface CanvasDimsAspect {
  aspectRatioId: AspectRatioId;
  aspectRatioIsLocked: boolean;
  aspectRatioValue: number;
}

const FREE_ASPECT: CanvasDimsAspect = { aspectRatioId: 'Free', aspectRatioIsLocked: false, aspectRatioValue: 1 };

/** How far a frame's ratio may drift from the locked one (whole-pixel rounding) and still hold the lock as it is. */
const RATIO_TOLERANCE = 0.01;

export interface CanvasDimsReconcileInput {
  /** The current generation frame, or `null` when the sync should stay inert (no canvas mode). */
  bbox: Bbox | null;
  /** The current committed generate width/height. */
  dims: { width: number; height: number };
  /** The current aspect lock; free when absent. */
  aspect?: CanvasDimsAspect;
  /** The bbox/generate snapping grid (model-derived; identical on both sides). */
  grid: number;
  /** The last snapshot this sync wrote/observed, or `null` on first run / after a reset. */
  prev: CanvasDimsSnapshot | null;
}

export type CanvasDimsReconcileResult =
  | { kind: 'none' }
  /**
   * The frame decides: snapped processing dimensions, plus the aspect fields when the frame's ratio is not the locked
   * one (a locked frame of another ratio moves the lock to it; an unlocked one stays free).
   */
  | ({ kind: 'patch-dims'; width: number; height: number } & Partial<CanvasDimsAspect>)
  /** Resize the bbox to the (grid-snapped) generate dims around its centre. */
  | { kind: 'set-bbox'; bbox: Bbox };

/**
 * The aspect fields a frame of `width`×`height` leaves Generate with; none when they already describe it. Only the
 * person's own lock (a chosen preset or the lock button, `aspectRatioIsLocked`) holds on the canvas: a preset label a
 * size alone implied (a new project's 1:1, a recalled size) would otherwise lock the frame nobody locked.
 */
const aspectForFrame = (aspect: CanvasDimsAspect, width: number, height: number): Partial<CanvasDimsAspect> => {
  const ratio = height > 0 ? width / height : 1;
  if (!aspect.aspectRatioIsLocked) {
    return aspect.aspectRatioId === 'Free' && aspect.aspectRatioIsLocked === false && aspect.aspectRatioValue === ratio
      ? {}
      : { ...FREE_ASPECT, aspectRatioValue: ratio };
  }
  if (
    aspect.aspectRatioValue > 0 &&
    Math.abs(ratio - aspect.aspectRatioValue) / aspect.aspectRatioValue <= RATIO_TOLERANCE
  ) {
    return {};
  }
  return { aspectRatioId: deriveAspectRatioId(width, height), aspectRatioIsLocked: true, aspectRatioValue: ratio };
};

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
  aspect = FREE_ASPECT,
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

    return { ...aspectForFrame(aspect, bbox.width, bbox.height), height, kind: 'patch-dims', width };
  }

  const dimsChanged = prev!.dimsWidth !== dims.width || prev!.dimsHeight !== dims.height;

  if (dimsChanged) {
    const width = clampDimension(dims.width, grid);
    const height = clampDimension(dims.height, grid);

    if (width === bbox.width && height === bbox.height) {
      return { kind: 'none' };
    }

    return { bbox: resizeBboxAroundCentre(bbox, { height, width }), kind: 'set-bbox' };
  }

  return { kind: 'none' };
};

/** The frame resized to `size` around its centre; the position rounds to whole pixels. */
export const resizeBboxAroundCentre = (bbox: Bbox, size: Pick<Bbox, 'height' | 'width'>): Bbox => ({
  height: size.height,
  width: size.width,
  // `+ 0` folds the -0 that rounding a small negative offset yields.
  x: Math.round(bbox.x + (bbox.width - size.width) / 2) + 0,
  y: Math.round(bbox.y + (bbox.height - size.height) / 2) + 0,
});

/** The minimal workbench store surface the sync depends on. */
export interface CanvasDimsSyncStore {
  commands: {
    canvas: Pick<WorkbenchCommands['canvas'], 'apply'>;
    generation: Pick<WorkbenchCommands['generation'], 'patchSettings'>;
    widgets: Pick<WorkbenchCommands['widgets'], 'patchValues'>;
  };
  getState(): WorkbenchState;
  subscribe(listener: () => void): () => void;
}

export interface CanvasDimsSync {
  dispose(): void;
}

/**
 * Records a frame resize as an undoable canvas step when the project's canvas engine is open (true), so canvas undo
 * puts the frame, and through this sync Generate's size, back; false leaves it to a `system` write.
 */
export type CanvasFrameCommit = (projectId: string, from: Bbox, to: Bbox) => boolean;

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

// External generators follow their own size policy, so the type selects the grid as the compiler's does.
const readModelType = (values: Record<string, unknown>): GenerateModelConfig['type'] => {
  const model = values.model;
  return model && typeof model === 'object' && (model as { type?: unknown }).type === 'external_image_generator'
    ? 'external_image_generator'
    : 'main';
};

export const createCanvasDimsSync = (
  store: CanvasDimsSyncStore,
  commitFrame: CanvasFrameCommit = () => false
): CanvasDimsSync => {
  let prev: CanvasDimsSnapshot | null = null;
  let lastProjectId: string | null = null;
  /** What the render size last saw of the active canvas project; null until it has seen one. */
  let renderPrev: {
    isExplicitlyLocked: boolean;
    method: CanvasScaleMethod;
    projectId: string;
    ratio: number | null;
  } | null = null;
  let isSyncing = false;

  /** Reconcile the frame with Generate's size. */
  const syncFrame = (): void => {
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

    const aspect: CanvasDimsAspect = {
      aspectRatioId:
        typeof generateValues.aspectRatioId === 'string' ? (generateValues.aspectRatioId as AspectRatioId) : 'Free',
      aspectRatioIsLocked: generateValues.aspectRatioIsLocked === true,
      aspectRatioValue:
        typeof generateValues.aspectRatioValue === 'number' && Number.isFinite(generateValues.aspectRatioValue)
          ? generateValues.aspectRatioValue
          : 1,
    };
    const result = reconcileCanvasDims({ aspect, bbox, dims: { height, width }, grid, prev });
    const projectId = project.id;

    switch (result.kind) {
      case 'none': {
        prev = createSnapshot(bbox, { height, width }, grid);
        return;
      }
      case 'patch-dims': {
        const { kind: _, ...patch } = result;
        prev = createSnapshot(bbox, result, grid);
        store.commands.generation.patchSettings(patch, projectId, 'system');
        return;
      }
      case 'set-bbox': {
        const nextBbox = result.bbox;
        prev = createSnapshot(nextBbox, nextBbox, grid);
        if (!commitFrame(projectId, bbox, nextBbox)) {
          store.commands.canvas.apply(projectId, { bbox: nextBbox, type: 'setCanvasBbox' }, 'system');
        }
        if (width !== nextBbox.width || height !== nextBbox.height) {
          store.commands.generation.patchSettings(
            {
              ...aspectForFrame(aspect, nextBbox.width, nextBbox.height),
              height: nextBbox.height,
              width: nextBbox.width,
            },
            projectId,
            'system'
          );
        }
        return;
      }
    }
  };

  // Acts only on a new locked ratio or on entering Custom, never on a size being typed: the Size section links those.
  const syncRenderSize = (): void => {
    const state = store.getState();
    const project = state.projects.find((candidate) => candidate.id === state.activeProjectId);

    if (!project || project.invocation.sourceId !== 'canvas') {
      renderPrev = null;
      return;
    }

    const generateValues = getProjectWidgetValues(project, 'generate');
    const base = readModelBase(generateValues);
    const variant = readModelVariant(generateValues);

    // Hold the change until the architecture answers, as the frame does: a fallback grid may be invalid for it.
    if (resolveModelGrid(base, variant) === null) {
      return;
    }

    const scaling = readCanvasScaling(getProjectWidgetValues(project, 'canvas'));
    const ratio = readLockedAspectRatio(generateValues);
    const isExplicitlyLocked = generateValues.aspectRatioIsLocked === true;
    const previous = renderPrev?.projectId === project.id ? renderPrev : null;
    renderPrev = { isExplicitlyLocked, method: scaling.method, projectId: project.id, ratio };

    if (
      !previous ||
      ratio === null ||
      scaling.method !== 'manual' ||
      // Locking a ratio the frame only labelled is a new lock, even though the ratio itself is unchanged.
      (previous.ratio === ratio &&
        previous.method === 'manual' &&
        (previous.isExplicitlyLocked || !isExplicitlyLocked)) ||
      // Without a size of its own, manual renders at the frame, which already holds the ratio.
      (scaling.width === null && scaling.height === null)
    ) {
      return;
    }

    const { bbox } = project.canvas.document;
    const model = base ? { base, type: readModelType(generateValues), variant } : undefined;
    const { grid } = getGenerationDimensions(model, (generateValues.pidMode as PidMode | undefined) ?? 'off');
    const fitted = fitCanvasScalingToRatio(
      { height: scaling.height ?? bbox.height, width: scaling.width ?? bbox.width },
      ratio,
      grid
    );

    if (fitted) {
      store.commands.widgets.patchValues(
        'canvas',
        { [CANVAS_SCALING_KEYS.height]: fitted.height, [CANVAS_SCALING_KEYS.width]: fitted.width },
        project.id,
        'system'
      );
    }
  };

  const handleChange = (): void => {
    // Dispatch synchronously re-enters this listener; skip the nested pass to bound writes.
    if (isSyncing) {
      return;
    }

    isSyncing = true;
    try {
      // After the frame, so a ratio a bbox edit just wrote reaches the render size in the same pass.
      syncFrame();
      syncRenderSize();
    } finally {
      isSyncing = false;
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
