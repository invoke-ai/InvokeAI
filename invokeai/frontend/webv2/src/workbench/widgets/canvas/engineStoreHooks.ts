/** Adapt engine-owned external stores to React here, preserving the engine's React-free boundary. */

import type {
  BboxToolOptions,
  BrushOptions,
  CanvasCoreStoreCapability,
  CanvasInteractionState,
  EraserOptions,
  GradientToolOptions,
  LassoToolOptions,
  MarqueeToolOptions,
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
 * Subscribe only to one layer's thumbnail version; null engines return undefined without subscribing for fallback
 * rendering.
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

/**
 * A primitive read of live engine state (document model, cached pixels), taken on every render and whenever the
 * engine's document changes or a layer publishes pixels. The React Compiler memoizes a plain render-time engine call
 * on its arguments, so it would keep a stale answer after a paint or another layer's edit; route such reads through
 * here.
 */
export const useCanvasEngineRead = <T extends boolean | number | string | null>(
  engine: CanvasCoreStoreCapability | null,
  read: () => T
): T => {
  const subscribe = useCallback(
    (listener: () => void) => {
      if (!engine) {
        return () => undefined;
      }
      const unsubscribePixels = engine.interaction.subscribe('layerPixelEpoch', listener);
      const unsubscribeDocument = engine.interaction.subscribe('documentEpoch', listener);
      return () => {
        unsubscribePixels();
        unsubscribeDocument();
      };
    },
    [engine]
  );
  return useSyncExternalStore(subscribe, read, read);
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

/** Bumped on every history-stack mutation; a history list re-reads the entries on change. */
export const useCanvasHistoryEpoch = (engine: CanvasCoreStoreCapability): number =>
  useCanvasInteractionState(engine, 'historyEpoch');

/** Whether a SAM/filter operation currently excludes ordinary canvas document edits. */
export const useCanvasDocumentEditingLocked = (engine: CanvasCoreStoreCapability | null): boolean => {
  const subscribe = useCallback(
    (listener: () => void) => engine?.interaction.subscribe('documentEditingLocked', listener) ?? (() => undefined),
    [engine]
  );
  const getSnapshot = useCallback(() => engine?.interaction.get('documentEditingLocked') ?? false, [engine]);
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
};

/** Re-renders when any live layer cache gains, loses, or changes pixels. */
export const useCanvasLayerPixelEpoch = (engine: CanvasCoreStoreCapability | null): number => {
  const subscribe = useCallback(
    (listener: () => void) => engine?.interaction.subscribe('layerPixelEpoch', listener) ?? (() => undefined),
    [engine]
  );
  const getSnapshot = useCallback(() => engine?.interaction.get('layerPixelEpoch') ?? 0, [engine]);
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
};

/** Read/write brush options directly through engine.interaction; no reducer mirror exists. */
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

/** Whether pixels are in flight as a floating selection (enables the transform bar while framed). */
export const useCanvasHasFloatingSelection = (engine: CanvasCoreStoreCapability): boolean =>
  useCanvasInteractionState(engine, 'hasFloatingSelection');

/** The lasso tool's current options (the committed boolean op mode). Write through `engine.interaction.set`. */
export const useLassoOptions = (engine: CanvasCoreStoreCapability): LassoToolOptions =>
  useCanvasInteractionState(engine, 'lassoOptions');

/** The marquee tool's current options (shape kind / boolean op mode). Write through `engine.interaction.set`. */
export const useMarqueeOptions = (engine: CanvasCoreStoreCapability): MarqueeToolOptions =>
  useCanvasInteractionState(engine, 'marqueeOptions');

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
