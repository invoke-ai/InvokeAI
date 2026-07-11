/**
 * React bindings for the engine's transient stores.
 *
 * `canvas-engine/engineStores.ts` deliberately ships React-free
 * (`useSyncExternalStore`-compatible) channels so the engine stays node-safe.
 * These hooks are the widget-side adapter — the one place React subscribes to
 * that engine-owned interaction state. Keeping the React import here (under
 * `widgets/`) preserves the engine's zero-React boundary.
 */

import type { CanvasOperationState } from '@workbench/canvas-engine/canvasOperationController';
import type { CanvasEngine } from '@workbench/canvas-engine/engine';
import type {
  BboxToolOptions,
  BrushOptions,
  EraserOptions,
  GradientToolOptions,
  FilterOperationSessionState,
  SamSessionSnapshot,
  LassoToolOptions,
  ScalarStore,
  ShapeToolOptions,
  TextEditSession,
  TextToolOptions,
  TransformSession,
  LayerThumbnailStatus,
} from '@workbench/canvas-engine/engineStores';
import type { ToolId } from '@workbench/canvas-engine/types';

import { useCallback, useSyncExternalStore } from 'react';

/** Subscribes the calling component to a single engine scalar store. */
const useScalarStore = <T>(store: ScalarStore<T>): T => useSyncExternalStore(store.subscribe, store.get);

/**
 * Subscribes to a single layer's thumbnail version on `engine`, re-rendering
 * only when that layer's cached pixels change (the engine bumps the version on
 * every repaint / rasterize). Tolerates a `null` engine — before the engine
 * mounts the hook simply reports `undefined` and never subscribes — so the
 * layers panel can render fallback thumbnails without an attached engine.
 */
export const useLayerThumbnailVersion = (engine: CanvasEngine | null, layerId: string): number | undefined => {
  const subscribe = useCallback(
    (onStoreChange: () => void) => engine?.stores.thumbnailVersion.subscribeKey(layerId, onStoreChange) ?? (() => {}),
    [engine, layerId]
  );
  const getSnapshot = useCallback(() => engine?.stores.thumbnailVersion.get(layerId), [engine, layerId]);
  return useSyncExternalStore(subscribe, getSnapshot);
};

/** Subscribes to one layer's thumbnail request state; an absent key is idle. */
export const useLayerThumbnailStatus = (
  engine: CanvasEngine | null,
  layerId: string
): LayerThumbnailStatus | 'idle' => {
  const subscribe = useCallback(
    (onStoreChange: () => void) => engine?.stores.thumbnailStatus.subscribeKey(layerId, onStoreChange) ?? (() => {}),
    [engine, layerId]
  );
  const getSnapshot = useCallback(() => engine?.stores.thumbnailStatus.get(layerId) ?? 'idle', [engine, layerId]);
  return useSyncExternalStore(subscribe, getSnapshot);
};

/** Current viewport zoom factor for `engine` (re-renders on zoom change). */
export const useCanvasZoom = (engine: CanvasEngine): number => useScalarStore(engine.stores.zoom);

/** Whether `engine` has render targets bound and its viewport is live. */
export const useCanvasViewportReady = (engine: CanvasEngine): boolean => useScalarStore(engine.stores.viewportReady);

/** The active tool id for `engine`. */
export const useCanvasActiveTool = (engine: CanvasEngine): ToolId => useScalarStore(engine.stores.activeTool);

const IDLE_CANVAS_OPERATION: CanvasOperationState = { status: 'idle' };

export const useCanvasOperation = (engine: CanvasEngine | null): CanvasOperationState => {
  const subscribe = useCallback(
    (listener: () => void) => engine?.canvasOperations.subscribe(listener) ?? (() => undefined),
    [engine]
  );
  const getSnapshot = useCallback(() => engine?.canvasOperations.getSnapshot() ?? IDLE_CANVAS_OPERATION, [engine]);
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
};

export const useSamSession = (engine: CanvasEngine): SamSessionSnapshot | null =>
  useScalarStore(engine.stores.samSession);

export const useFilterSession = (engine: CanvasEngine): FilterOperationSessionState | null =>
  useScalarStore(engine.stores.filterSession);

/** Whether the engine-owned canvas history has an entry to undo (enables the header undo button). */
export const useCanvasCanUndo = (engine: CanvasEngine): boolean => useScalarStore(engine.stores.canUndo);

/** Whether the engine-owned canvas history has an entry to redo (enables the header redo button). */
export const useCanvasCanRedo = (engine: CanvasEngine): boolean => useScalarStore(engine.stores.canRedo);

/**
 * The brush tool's current options (size / color / opacity / pressure). Write
 * through `engine.stores.brushOptions.set(...)` directly — there is no reducer
 * mirror to dispatch through, so the options bar reads and writes this one
 * store.
 */
export const useBrushOptions = (engine: CanvasEngine): BrushOptions => useScalarStore(engine.stores.brushOptions);

/** The eraser tool's current options (size / opacity). Write through `engine.stores.eraserOptions.set(...)`. */
export const useEraserOptions = (engine: CanvasEngine): EraserOptions => useScalarStore(engine.stores.eraserOptions);

/** The bbox tool's current options (aspect lock / ratio). Write through `engine.stores.bboxOptions.set(...)`. */
export const useBboxOptions = (engine: CanvasEngine): BboxToolOptions => useScalarStore(engine.stores.bboxOptions);

/** The bbox tool's current snapping grid size (document px). */
export const useBboxGrid = (engine: CanvasEngine): number => useScalarStore(engine.stores.bboxGrid);

/** The active transform-tool session (layer id + live transform), or `null`. */
export const useTransformSession = (engine: CanvasEngine): TransformSession | null =>
  useScalarStore(engine.stores.transformSession);

/** Whether a pixel selection currently exists (enables fill/erase/invert/deselect controls). */
export const useCanvasHasSelection = (engine: CanvasEngine): boolean => useScalarStore(engine.stores.hasSelection);

/** The lasso tool's current options (the committed boolean op mode). Write through `engine.stores.lassoOptions.set(...)`. */
export const useLassoOptions = (engine: CanvasEngine): LassoToolOptions => useScalarStore(engine.stores.lassoOptions);

/** The shape tool's current options (kind / fill / stroke / stroke width). Write through `engine.stores.shapeOptions.set(...)`. */
export const useShapeOptions = (engine: CanvasEngine): ShapeToolOptions => useScalarStore(engine.stores.shapeOptions);

/** The gradient tool's current options (kind / angle / stops). Write through `engine.stores.gradientOptions.set(...)`. */
export const useGradientOptions = (engine: CanvasEngine): GradientToolOptions =>
  useScalarStore(engine.stores.gradientOptions);

/** The text tool's current options (font / size / weight / line-height / align / color). */
export const useTextOptions = (engine: CanvasEngine): TextToolOptions => useScalarStore(engine.stores.textOptions);

/** The active text-editing session (create/edit mode + live source + transform), or `null`. */
export const useTextEditSession = (engine: CanvasEngine): TextEditSession | null =>
  useScalarStore(engine.stores.textEditSession);
