import { $alt, $ctrl, $meta, $shift } from '@invoke-ai/ui-library';
import type { Store } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import { $isDebugging } from 'app/store/nanostores/isDebugging';
import type { RootState } from 'app/store/store';
import { EntityToKonvaMap } from 'features/controlLayers/konva/entityToKonvaMap';
import { setStageEventHandlers } from 'features/controlLayers/konva/events';
import { arrangeEntities } from 'features/controlLayers/konva/renderers/arrange';
import { renderBackgroundLayer } from 'features/controlLayers/konva/renderers/background';
import { updateBboxes } from 'features/controlLayers/konva/renderers/bbox';
import { renderControlAdapters } from 'features/controlLayers/konva/renderers/controlAdapters';
import { renderLayers } from 'features/controlLayers/konva/renderers/layers';
import {
  renderBboxPreview,
  renderDocumentBoundsOverlay,
  scaleToolPreview,
} from 'features/controlLayers/konva/renderers/preview';
import { renderRegions } from 'features/controlLayers/konva/renderers/regions';
import { fitDocumentToStage } from 'features/controlLayers/konva/renderers/stage';
import {
  $stageAttrs,
  bboxChanged,
  brushWidthChanged,
  caBboxChanged,
  caTranslated,
  eraserWidthChanged,
  layerBboxChanged,
  layerBrushLineAdded,
  layerEraserLineAdded,
  layerLinePointAdded,
  layerRectAdded,
  layerTranslated,
  rgBboxChanged,
  rgBrushLineAdded,
  rgEraserLineAdded,
  rgLinePointAdded,
  rgRectAdded,
  rgTranslated,
  toolBufferChanged,
  toolChanged,
} from 'features/controlLayers/store/canvasV2Slice';
import type {
  BboxChangedArg,
  BrushLineAddedArg,
  CanvasEntity,
  CanvasV2State,
  EraserLineAddedArg,
  PointAddedToLineArg,
  PosChangedArg,
  RectShapeAddedArg,
  Tool,
} from 'features/controlLayers/store/types';
import type Konva from 'konva';
import type { IRect, Vector2d } from 'konva/lib/types';
import { debounce } from 'lodash-es';
import type { RgbaColor } from 'react-colorful';
import { getImageDTO } from 'services/api/endpoints/images';

/**
 * Initializes the canvas renderer. It subscribes to the redux store and listens for changes directly, bypassing the
 * react rendering cycle entirely, improving canvas performance.
 * @param store The Redux store
 * @param stage The Konva stage
 * @param container The stage's target container element
 * @returns A cleanup function
 */
export const initializeRenderer = (
  store: Store<RootState>,
  stage: Konva.Stage,
  container: HTMLDivElement | null
): (() => void) => {
  const _log = logger('konva');
  /**
   * Logs a message to the console if debugging is enabled.
   */
  const logIfDebugging = (message: string) => {
    if ($isDebugging.get()) {
      _log.trace(message);
    }
  };

  logIfDebugging('Initializing renderer');
  if (!container) {
    // Nothing to clean up
    logIfDebugging('No stage container, skipping initialization');
    return () => {};
  }

  stage.container(container);

  // Set up callbacks for various events
  const onPosChanged = (arg: PosChangedArg, entityType: CanvasEntity['type']) => {
    logIfDebugging('Position changed');
    if (entityType === 'layer') {
      dispatch(layerTranslated(arg));
    } else if (entityType === 'control_adapter') {
      dispatch(caTranslated(arg));
    } else if (entityType === 'regional_guidance') {
      dispatch(rgTranslated(arg));
    }
  };
  const onBboxChanged = (arg: BboxChangedArg, entityType: CanvasEntity['type']) => {
    logIfDebugging('Entity bbox changed');
    if (entityType === 'layer') {
      dispatch(layerBboxChanged(arg));
    } else if (entityType === 'control_adapter') {
      dispatch(caBboxChanged(arg));
    } else if (entityType === 'regional_guidance') {
      dispatch(rgBboxChanged(arg));
    }
  };
  const onBrushLineAdded = (arg: BrushLineAddedArg, entityType: CanvasEntity['type']) => {
    logIfDebugging('Brush line added');
    if (entityType === 'layer') {
      dispatch(layerBrushLineAdded(arg));
    } else if (entityType === 'regional_guidance') {
      dispatch(rgBrushLineAdded(arg));
    }
  };
  const onEraserLineAdded = (arg: EraserLineAddedArg, entityType: CanvasEntity['type']) => {
    logIfDebugging('Eraser line added');
    if (entityType === 'layer') {
      dispatch(layerEraserLineAdded(arg));
    } else if (entityType === 'regional_guidance') {
      dispatch(rgEraserLineAdded(arg));
    }
  };
  const onPointAddedToLine = (arg: PointAddedToLineArg, entityType: CanvasEntity['type']) => {
    logIfDebugging('Point added to line');
    if (entityType === 'layer') {
      dispatch(layerLinePointAdded(arg));
    } else if (entityType === 'regional_guidance') {
      dispatch(rgLinePointAdded(arg));
    }
  };
  const onRectShapeAdded = (arg: RectShapeAddedArg, entityType: CanvasEntity['type']) => {
    logIfDebugging('Rect shape added');
    if (entityType === 'layer') {
      dispatch(layerRectAdded(arg));
    } else if (entityType === 'regional_guidance') {
      dispatch(rgRectAdded(arg));
    }
  };
  const onBboxTransformed = (bbox: IRect) => {
    logIfDebugging('Generation bbox transformed');
    dispatch(bboxChanged(bbox));
  };
  const onBrushWidthChanged = (width: number) => {
    logIfDebugging('Brush width changed');
    dispatch(brushWidthChanged(width));
  };
  const onEraserWidthChanged = (width: number) => {
    logIfDebugging('Eraser width changed');
    dispatch(eraserWidthChanged(width));
  };
  const setTool = (tool: Tool) => {
    logIfDebugging('Tool selection changed');
    dispatch(toolChanged(tool));
  };
  const setToolBuffer = (toolBuffer: Tool | null) => {
    logIfDebugging('Tool buffer changed');
    dispatch(toolBufferChanged(toolBuffer));
  };

  const _getSelectedEntity = (canvasV2: CanvasV2State): CanvasEntity | null => {
    const identifier = canvasV2.selectedEntityIdentifier;
    let selectedEntity: CanvasEntity | null = null;
    if (!identifier) {
      selectedEntity = null;
    } else if (identifier.type === 'layer') {
      selectedEntity = canvasV2.layers.find((i) => i.id === identifier.id) ?? null;
    } else if (identifier.type === 'control_adapter') {
      selectedEntity = canvasV2.controlAdapters.find((i) => i.id === identifier.id) ?? null;
    } else if (identifier.type === 'ip_adapter') {
      selectedEntity = canvasV2.ipAdapters.find((i) => i.id === identifier.id) ?? null;
    } else if (identifier.type === 'regional_guidance') {
      selectedEntity = canvasV2.regions.find((i) => i.id === identifier.id) ?? null;
    } else {
      selectedEntity = null;
    }
    logIfDebugging('Selected entity changed');
    return selectedEntity;
  };

  const _getCurrentFill = (canvasV2: CanvasV2State, selectedEntity: CanvasEntity | null) => {
    let currentFill: RgbaColor = canvasV2.tool.fill;
    if (selectedEntity && selectedEntity.type === 'regional_guidance') {
      currentFill = { ...selectedEntity.fill, a: canvasV2.settings.maskOpacity };
    } else {
      currentFill = canvasV2.tool.fill;
    }
    logIfDebugging('Current fill changed');
    return currentFill;
  };

  const { getState, subscribe, dispatch } = store;

  // On the first render, we need to render everything.
  let isFirstRender = true;

  // Stage interaction listeners need helpers to get and update current state. Some of the state is read-only, like
  // bbox, document and tool state, while interaction state is read-write.

  // Read-only state, derived from redux
  let prevCanvasV2 = getState().canvasV2;
  let prevSelectedEntity: CanvasEntity | null = _getSelectedEntity(prevCanvasV2);
  let prevCurrentFill: RgbaColor = _getCurrentFill(prevCanvasV2, prevSelectedEntity);
  const getSelectedEntity = () => prevSelectedEntity;
  const getCurrentFill = () => prevCurrentFill;
  const getBbox = () => getState().canvasV2.bbox;
  const getDocument = () => getState().canvasV2.document;
  const getToolState = () => getState().canvasV2.tool;

  // Read-write state, ephemeral interaction state
  let isDrawing = false;
  const getIsDrawing = () => isDrawing;
  const setIsDrawing = (val: boolean) => {
    isDrawing = val;
  };

  let isMouseDown = false;
  const getIsMouseDown = () => isMouseDown;
  const setIsMouseDown = (val: boolean) => {
    isMouseDown = val;
  };

  let lastAddedPoint: Vector2d | null = null;
  const getLastAddedPoint = () => lastAddedPoint;
  const setLastAddedPoint = (val: Vector2d | null) => {
    lastAddedPoint = val;
  };

  let lastMouseDownPos: Vector2d | null = null;
  const getLastMouseDownPos = () => lastMouseDownPos;
  const setLastMouseDownPos = (val: Vector2d | null) => {
    lastMouseDownPos = val;
  };

  let lastCursorPos: Vector2d | null = null;
  const getLastCursorPos = () => lastCursorPos;
  const setLastCursorPos = (val: Vector2d | null) => {
    lastCursorPos = val;
  };

  let spaceKey = false;
  const getSpaceKey = () => spaceKey;
  const setSpaceKey = (val: boolean) => {
    spaceKey = val;
  };

  const cleanupListeners = setStageEventHandlers({
    stage,
    getToolState,
    setTool,
    setToolBuffer,
    getIsDrawing,
    setIsDrawing,
    getIsMouseDown,
    setIsMouseDown,
    getSelectedEntity,
    getLastAddedPoint,
    setLastAddedPoint,
    getLastCursorPos,
    setLastCursorPos,
    getLastMouseDownPos,
    setLastMouseDownPos,
    getSpaceKey,
    setSpaceKey,
    setStageAttrs: $stageAttrs.set,
    getDocument,
    getBbox,
    onBrushLineAdded,
    onEraserLineAdded,
    onPointAddedToLine,
    onRectShapeAdded,
    onBrushWidthChanged,
    onEraserWidthChanged,
    getCurrentFill,
  });

  // Calculating bounding boxes is expensive, must be debounced to not block the UI thread during a user interaction.
  // TODO(psyche): Figure out how to do this in a worker. Probably means running the renderer in a worker and sending
  // the entire state over when needed.
  const debouncedUpdateBboxes = debounce(updateBboxes, 300);

  const regionMap = new EntityToKonvaMap();
  const layerMap = new EntityToKonvaMap();
  const controlAdapterMap = new EntityToKonvaMap();

  const renderCanvas = () => {
    const { canvasV2 } = store.getState();

    if (prevCanvasV2 === canvasV2 && !isFirstRender) {
      logIfDebugging('No changes detected, skipping render');
      return;
    }

    const selectedEntity = _getSelectedEntity(canvasV2);
    const currentFill = _getCurrentFill(canvasV2, selectedEntity);

    if (
      isFirstRender ||
      canvasV2.layers !== prevCanvasV2.layers ||
      canvasV2.tool.selected !== prevCanvasV2.tool.selected
    ) {
      logIfDebugging('Rendering layers');
      renderLayers(stage, layerMap, canvasV2.layers, canvasV2.tool.selected, onPosChanged);
    }

    if (
      isFirstRender ||
      canvasV2.regions !== prevCanvasV2.regions ||
      canvasV2.settings.maskOpacity !== prevCanvasV2.settings.maskOpacity ||
      canvasV2.tool.selected !== prevCanvasV2.tool.selected
    ) {
      logIfDebugging('Rendering regions');
      renderRegions(
        stage,
        regionMap,
        canvasV2.regions,
        canvasV2.settings.maskOpacity,
        canvasV2.tool.selected,
        canvasV2.selectedEntityIdentifier,
        onPosChanged
      );
    }

    if (isFirstRender || canvasV2.controlAdapters !== prevCanvasV2.controlAdapters) {
      logIfDebugging('Rendering control adapters');
      renderControlAdapters(stage, controlAdapterMap, canvasV2.controlAdapters, getImageDTO);
    }

    if (isFirstRender || canvasV2.document !== prevCanvasV2.document) {
      logIfDebugging('Rendering document bounds overlay');
      renderDocumentBoundsOverlay(stage, getDocument);
    }

    if (isFirstRender || canvasV2.bbox !== prevCanvasV2.bbox || canvasV2.tool.selected !== prevCanvasV2.tool.selected) {
      logIfDebugging('Rendering generation bbox');
      renderBboxPreview(
        stage,
        canvasV2.bbox,
        canvasV2.tool.selected,
        getBbox,
        onBboxTransformed,
        $shift.get,
        $ctrl.get,
        $meta.get,
        $alt.get
      );
    }

    if (
      isFirstRender ||
      canvasV2.layers !== prevCanvasV2.layers ||
      canvasV2.controlAdapters !== prevCanvasV2.controlAdapters ||
      canvasV2.regions !== prevCanvasV2.regions
    ) {
      logIfDebugging('Updating entity bboxes');
      debouncedUpdateBboxes(stage, canvasV2.layers, canvasV2.controlAdapters, canvasV2.regions, onBboxChanged);
    }

    if (
      isFirstRender ||
      canvasV2.layers !== prevCanvasV2.layers ||
      canvasV2.controlAdapters !== prevCanvasV2.controlAdapters ||
      canvasV2.regions !== prevCanvasV2.regions
    ) {
      logIfDebugging('Arranging entities');
      arrangeEntities(stage, canvasV2.layers, canvasV2.controlAdapters, canvasV2.regions);
    }

    prevCanvasV2 = canvasV2;
    prevSelectedEntity = selectedEntity;
    prevCurrentFill = currentFill;

    if (isFirstRender) {
      isFirstRender = false;
    }
  };

  // We can use a resize observer to ensure the stage always fits the container. We also need to re-render the bg and
  // document bounds overlay when the stage is resized.
  const fitStageToContainer = () => {
    stage.width(container.offsetWidth);
    stage.height(container.offsetHeight);
    $stageAttrs.set({
      x: stage.x(),
      y: stage.y(),
      width: stage.width(),
      height: stage.height(),
      scale: stage.scaleX(),
    });
    renderBackgroundLayer(stage);
    renderDocumentBoundsOverlay(stage, getDocument);
  };

  const resizeObserver = new ResizeObserver(fitStageToContainer);
  resizeObserver.observe(container);
  fitStageToContainer();

  const unsubscribeRenderer = subscribe(renderCanvas);

  logIfDebugging('First render of konva stage');
  // On first render, the document should be fit to the stage.
  const stageAttrs = fitDocumentToStage(stage, prevCanvasV2.document);
  // The HUD displays some of the stage attributes, so we need to update it here.
  $stageAttrs.set(stageAttrs);
  scaleToolPreview(stage, getToolState());
  renderCanvas();

  return () => {
    logIfDebugging('Cleaning up konva renderer');
    unsubscribeRenderer();
    cleanupListeners();
    resizeObserver.disconnect();
  };
};
