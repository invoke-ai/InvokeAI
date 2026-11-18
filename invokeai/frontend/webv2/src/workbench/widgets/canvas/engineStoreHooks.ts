/** Adapt engine-owned external stores to React here, preserving the engine's React-free boundary. */

import type {
  BboxToolOptions,
  BrushOptions,
  CanvasCoreStoreCapability,
  CanvasHistoryCapability,
  CanvasHistoryEntries,
  CanvasInteractionState,
  EraserOptions,
  GradientToolOptions,
  LassoToolOptions,
  LayerQuickSwitchState,
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
  IDLE_CONTROL_DETECTION_STATUS,
  type CanvasOperationState,
  type ControlDetectionStatus,
  type FilterOperationSessionState,
  type SamSessionSnapshot,
} from '@workbench/canvas-operations/api';
import { useCallback, useSyncExternalStore } from 'react';

const EMPTY_LAYER_QUICK_SWITCH: LayerQuickSwitchState = { bookmarkedLayerId: null, previousLayerId: null };

const useCanvasInteractionState = <K extends keyof CanvasInteractionState>(
  engine: CanvasCoreStoreCapability,
  key: K
): CanvasInteractionState[K] => {
  const subscribe = useCallback((listener: () => void) => engine.interaction.subscribe(key, listener), [engine, key]);
  const getSnapshot = useCallback(() => engine.interaction.get(key), [engine, key]);
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
};

/** Subscribes to one layer's thumbnail version; a null engine returns undefined without subscribing. */
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

/** Render-time engine reads: the React Compiler would memoize a plain call on its args and go stale. */
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

/**
 * A primitive read of the engine's document model, as `useCanvasEngineRead` but re-read only when the document
 * changes: for answers pixels cannot change, such as command eligibility, so painting never re-runs them.
 */
export const useCanvasDocumentRead = <T extends boolean | number | string | null>(
  engine: CanvasCoreStoreCapability | null,
  read: () => T
): T => {
  const subscribe = useCallback(
    (listener: () => void) => engine?.interaction.subscribe('documentEpoch', listener) ?? (() => undefined),
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

export const useCanvasZoom = (engine: CanvasCoreStoreCapability): number => useCanvasInteractionState(engine, 'zoom');

/** Whether `engine` has render targets bound and its viewport is live. */
export const useCanvasViewportReady = (engine: CanvasCoreStoreCapability): boolean =>
  useCanvasInteractionState(engine, 'viewportReady');

/** The control layer the canvas shows as its original (or original under map); null while every map shows. */
export const useControlLayerView = (engine: CanvasCoreStoreCapability): CanvasInteractionState['controlLayerView'] =>
  useCanvasInteractionState(engine, 'controlLayerView');

/** Whether a staged or progress preview, shown or toggled off, hides control layers, masks and regions. */
export const useCanvasStagingHidesOverlays = (engine: CanvasCoreStoreCapability): boolean =>
  useCanvasInteractionState(engine, 'stagingHidesOverlays');

/** The generation frame while a Frame tool drag reshapes it; null when the committed frame is current. */
export const useCanvasBboxPreview = (engine: CanvasCoreStoreCapability): CanvasInteractionState['bboxPreview'] =>
  useCanvasInteractionState(engine, 'bboxPreview');

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

/** One control layer's background detection status; a null engine reads as idle without subscribing. */
export const useControlDetectionStatus = (engine: object | null, layerId: string): ControlDetectionStatus => {
  const subscribe = useCallback(
    (listener: () => void) =>
      engine ? getCanvasOperations(engine).subscribeControlDetectionStatus(layerId, listener) : () => undefined,
    [engine, layerId]
  );
  const getSnapshot = useCallback(
    () => (engine ? getCanvasOperations(engine).getControlDetectionStatus(layerId) : IDLE_CONTROL_DETECTION_STATUS),
    [engine, layerId]
  );
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
};

export const useCanvasCanUndo = (engine: CanvasCoreStoreCapability): boolean =>
  useCanvasInteractionState(engine, 'canUndo');

export const useCanvasCanRedo = (engine: CanvasCoreStoreCapability): boolean =>
  useCanvasInteractionState(engine, 'canRedo');

type CanvasHistoryLabelEngine = CanvasCoreStoreCapability & {
  readonly history: Pick<CanvasHistoryCapability, 'getEntries'>;
};

const readUndoLabel = (entries: CanvasHistoryEntries): string | undefined => entries.past.at(-1);
const readRedoLabel = (entries: CanvasHistoryEntries): string | undefined => entries.future[0];

/** One history entry's label, re-read on every history-stack mutation; null when there is none. */
const useCanvasHistoryLabel = (
  engine: CanvasHistoryLabelEngine,
  read: (entries: CanvasHistoryEntries) => string | undefined
): string | null => {
  const subscribe = useCallback(
    (listener: () => void) => engine.interaction.subscribe('historyEpoch', listener),
    [engine]
  );
  const getSnapshot = useCallback(() => read(engine.history.getEntries()) ?? null, [engine, read]);
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
};

/** The label of the step `undo()` would revert, or null with nothing to undo. */
export const useCanvasUndoLabel = (engine: CanvasHistoryLabelEngine): string | null =>
  useCanvasHistoryLabel(engine, readUndoLabel);

/** The label of the step `redo()` would replay, or null with nothing to redo. */
export const useCanvasRedoLabel = (engine: CanvasHistoryLabelEngine): string | null =>
  useCanvasHistoryLabel(engine, readRedoLabel);

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

/** Read/write brush options directly through engine.interaction; no reducer mirror exists. */
export const useBrushOptions = (engine: CanvasCoreStoreCapability): BrushOptions =>
  useCanvasInteractionState(engine, 'brushOptions');

export const useEraserOptions = (engine: CanvasCoreStoreCapability): EraserOptions =>
  useCanvasInteractionState(engine, 'eraserOptions');

/** The Q quick-switch memory; a null engine reads as empty without subscribing. */
export const useLayerQuickSwitch = (engine: CanvasCoreStoreCapability | null): LayerQuickSwitchState => {
  const subscribe = useCallback(
    (listener: () => void) => engine?.interaction.subscribe('layerQuickSwitch', listener) ?? (() => undefined),
    [engine]
  );
  const getSnapshot = useCallback(
    () => engine?.interaction.get('layerQuickSwitch') ?? EMPTY_LAYER_QUICK_SWITCH,
    [engine]
  );
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
};

export const useBboxOptions = (engine: CanvasCoreStoreCapability): BboxToolOptions =>
  useCanvasInteractionState(engine, 'bboxOptions');

/** The bbox tool's current snapping grid size (document px). */
export const useBboxGrid = (engine: CanvasCoreStoreCapability): number => useCanvasInteractionState(engine, 'bboxGrid');

export const useTransformSession = (engine: CanvasCoreStoreCapability): TransformSession | null =>
  useCanvasInteractionState(engine, 'transformSession');

/** The layer a requested transform waits on until its stroke saves; null when nothing is waiting. */
export const usePendingTransformLayerId = (engine: CanvasCoreStoreCapability): string | null =>
  useCanvasInteractionState(engine, 'pendingTransformLayerId');

/** Whether a pixel selection currently exists (enables fill/erase/invert/deselect controls). */
export const useCanvasHasSelection = (engine: CanvasCoreStoreCapability): boolean =>
  useCanvasInteractionState(engine, 'hasSelection');

/** Whether pixels are in flight as a floating selection (enables the transform bar while framed). */
export const useCanvasHasFloatingSelection = (engine: CanvasCoreStoreCapability): boolean =>
  useCanvasInteractionState(engine, 'hasFloatingSelection');

export const useLassoOptions = (engine: CanvasCoreStoreCapability): LassoToolOptions =>
  useCanvasInteractionState(engine, 'lassoOptions');

export const useMarqueeOptions = (engine: CanvasCoreStoreCapability): MarqueeToolOptions =>
  useCanvasInteractionState(engine, 'marqueeOptions');

export const useShapeOptions = (engine: CanvasCoreStoreCapability): ShapeToolOptions =>
  useCanvasInteractionState(engine, 'shapeOptions');

export const useGradientOptions = (engine: CanvasCoreStoreCapability): GradientToolOptions =>
  useCanvasInteractionState(engine, 'gradientOptions');

export const useTextOptions = (engine: CanvasCoreStoreCapability): TextToolOptions =>
  useCanvasInteractionState(engine, 'textOptions');

export const useTextEditSession = (engine: CanvasCoreStoreCapability): TextEditSession | null =>
  useCanvasInteractionState(engine, 'textEditSession');

export const useTransformOptions = (engine: CanvasCoreStoreCapability) =>
  useCanvasInteractionState(engine, 'transformOptions');

/** PSD preparation/worker/download lifecycle belongs to the engine, across widget remounts. */
export const usePsdExportStatus = (
  engine: CanvasCoreStoreCapability | null
): CanvasInteractionState['psdExportStatus'] => {
  const subscribe = useCallback(
    (listener: () => void) => engine?.interaction.subscribe('psdExportStatus', listener) ?? (() => undefined),
    [engine]
  );
  const getSnapshot = useCallback(() => engine?.interaction.get('psdExportStatus') ?? 'idle', [engine]);
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
};
