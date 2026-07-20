/**
 * React bindings for the engine's transient stores.
 *
 * `canvas-engine/engineStores.ts` deliberately ships React-free
 * (`useSyncExternalStore`-compatible) channels so the engine stays node-safe.
 * These hooks are the widget-side adapter — the one place React subscribes to
 * that engine-owned interaction state. Keeping the React import here (under
 * `widgets/`) preserves the engine's zero-React boundary.
 */

import type {
  BboxToolOptions,
  BrushOptions,
  CanvasCoreStoreCapability,
  CanvasInteractionState,
  EraserOptions,
  GradientToolOptions,
  LassoToolOptions,
  ShapeToolOptions,
  TextEditSession,
  TextToolOptions,
  TransformSession,
  LayerThumbnailStatus,
  ToolId,
} from '@workbench/canvas-engine/api';

import {
  getCanvasOperations,
  type CanvasOperationState,
  type FilterOperationSessionState,
  type SamSessionSnapshot,
} from '@workbench/canvas-operations/api';
import { useCallback, useSyncExternalStore } from 'react';

/** Subscribes the calling component to a single engine scalar store. */
const useCanvasInteractionState = <K extends keyof CanvasInteractionState>(
  engine: CanvasCoreStoreCapability,
  key: K
): CanvasInteractionState[K] => {
  const subscribe = useCallback((listener: () => void) => engine.interaction.subscribe(key, listener), [engine, key]);
  const getSnapshot = useCallback(() => engine.interaction.get(key), [engine, key]);
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
};

/**
 * Subscribes to a single layer's thumbnail version on `engine`, re-rendering
 * only when that layer's cached pixels change (the engine bumps the version on
 * every repaint / rasterize). Tolerates a `null` engine — before the engine
 * mounts the hook simply reports `undefined` and never subscribes — so the
 * layers panel can render fallback thumbnails without an attached engine.
 */
export const useLayerThumbnailVersion = (
  engine: CanvasCoreStoreCapability | null,
  layerId: string
): number | undefined => {
  const subscribe = useCallback(
    (onStoreChange: () => void) =>
      engine?.interaction.subscribeLayerThumbnailVersion(layerId, onStoreChange) ?? (() => {}),
    [engine, layerId]
  );
  const getSnapshot = useCallback(() => engine?.interaction.getLayerThumbnailVersion(layerId), [engine, layerId]);
  return useSyncExternalStore(subscribe, getSnapshot);
};

/** Subscribes to one layer's thumbnail request state; an absent key is idle. */
export const useLayerThumbnailStatus = (
  engine: CanvasCoreStoreCapability | null,
  layerId: string
): LayerThumbnailStatus | 'idle' => {
  const subscribe = useCallback(
    (onStoreChange: () => void) =>
      engine?.interaction.subscribeLayerThumbnailStatus(layerId, onStoreChange) ?? (() => {}),
    [engine, layerId]
  );
  const getSnapshot = useCallback(
    () => engine?.interaction.getLayerThumbnailStatus(layerId) ?? 'idle',
    [engine, layerId]
  );
  return useSyncExternalStore(subscribe, getSnapshot);
};

/** Current viewport zoom factor for `engine` (re-renders on zoom change). */
export const useCanvasZoom = (engine: CanvasCoreStoreCapability): number => useCanvasInteractionState(engine, 'zoom');

/** Whether `engine` has render targets bound and its viewport is live. */
export const useCanvasViewportReady = (engine: CanvasCoreStoreCapability): boolean =>
  useCanvasInteractionState(engine, 'viewportReady');

/** The active tool id for `engine`. */
export const useCanvasActiveTool = (engine: CanvasCoreStoreCapability): ToolId =>
  useCanvasInteractionState(engine, 'activeTool');

const IDLE_CANVAS_OPERATION: CanvasOperationState = { status: 'idle' };

export const useCanvasOperation = (engine: object | null): CanvasOperationState => {
  const subscribe = useCallback(
    (listener: () => void) => (engine ? getCanvasOperations(engine).subscribeOperation(listener) : () => undefined),
    [engine]
  );
  const getSnapshot = useCallback(
    () => (engine ? getCanvasOperations(engine).getOperationState() : IDLE_CANVAS_OPERATION),
    [engine]
  );
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
};

export const useSamSession = (engine: object): SamSessionSnapshot | null => {
  const operations = getCanvasOperations(engine);
  return useSyncExternalStore(
    operations.subscribeSamSession,
    operations.getSamSessionState,
    operations.getSamSessionState
  );
};

export const useFilterSession = (engine: object): FilterOperationSessionState | null => {
  const operations = getCanvasOperations(engine);
  return useSyncExternalStore(
    operations.subscribeFilterSession,
    operations.getFilterSessionState,
    operations.getFilterSessionState
  );
};

/** Whether the engine-owned canvas history has an entry to undo (enables the header undo button). */
export const useCanvasCanUndo = (engine: CanvasCoreStoreCapability): boolean =>
  useCanvasInteractionState(engine, 'canUndo');

/** Whether the engine-owned canvas history has an entry to redo (enables the header redo button). */
export const useCanvasCanRedo = (engine: CanvasCoreStoreCapability): boolean =>
  useCanvasInteractionState(engine, 'canRedo');

/** Whether a SAM/filter operation currently excludes ordinary canvas document edits. */
export const useCanvasDocumentEditingLocked = (engine: CanvasCoreStoreCapability | null): boolean => {
  const subscribe = useCallback(
    (listener: () => void) => engine?.interaction.subscribe('documentEditingLocked', listener) ?? (() => undefined),
    [engine]
  );
  const getSnapshot = useCallback(() => engine?.interaction.get('documentEditingLocked') ?? false, [engine]);
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
};

/**
 * The brush tool's current options (size / color / opacity / pressure). Write
 * through `engine.interaction.set('brushOptions', ...)` directly — there is no reducer
 * mirror to dispatch through, so the options bar reads and writes this one
 * store.
 */
export const useBrushOptions = (engine: CanvasCoreStoreCapability): BrushOptions =>
  useCanvasInteractionState(engine, 'brushOptions');

/** The eraser tool's current options (size / opacity). Write through `engine.interaction.set`. */
export const useEraserOptions = (engine: CanvasCoreStoreCapability): EraserOptions =>
  useCanvasInteractionState(engine, 'eraserOptions');

/** The bbox tool's current options (aspect lock / ratio). Write through `engine.interaction.set`. */
export const useBboxOptions = (engine: CanvasCoreStoreCapability): BboxToolOptions =>
  useCanvasInteractionState(engine, 'bboxOptions');

/** The bbox tool's current snapping grid size (document px). */
export const useBboxGrid = (engine: CanvasCoreStoreCapability): number => useCanvasInteractionState(engine, 'bboxGrid');

/** The active transform-tool session (layer id + live transform), or `null`. */
export const useTransformSession = (engine: CanvasCoreStoreCapability): TransformSession | null =>
  useCanvasInteractionState(engine, 'transformSession');

/** Whether a pixel selection currently exists (enables fill/erase/invert/deselect controls). */
export const useCanvasHasSelection = (engine: CanvasCoreStoreCapability): boolean =>
  useCanvasInteractionState(engine, 'hasSelection');

/** The lasso tool's current options (the committed boolean op mode). Write through `engine.interaction.set`. */
export const useLassoOptions = (engine: CanvasCoreStoreCapability): LassoToolOptions =>
  useCanvasInteractionState(engine, 'lassoOptions');

/** The shape tool's current options (kind / fill / stroke / stroke width). Write through `engine.interaction.set`. */
export const useShapeOptions = (engine: CanvasCoreStoreCapability): ShapeToolOptions =>
  useCanvasInteractionState(engine, 'shapeOptions');

/** The gradient tool's current options (kind / angle / stops). Write through `engine.interaction.set`. */
export const useGradientOptions = (engine: CanvasCoreStoreCapability): GradientToolOptions =>
  useCanvasInteractionState(engine, 'gradientOptions');

/** The text tool's current options (font / size / weight / line-height / align / color). */
export const useTextOptions = (engine: CanvasCoreStoreCapability): TextToolOptions =>
  useCanvasInteractionState(engine, 'textOptions');

/** The active text-editing session (create/edit mode + live source + transform), or `null`. */
export const useTextEditSession = (engine: CanvasCoreStoreCapability): TextEditSession | null =>
  useCanvasInteractionState(engine, 'textEditSession');
