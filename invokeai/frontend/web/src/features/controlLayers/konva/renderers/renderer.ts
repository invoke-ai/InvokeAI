import { $alt, $ctrl, $meta, $shift } from '@invoke-ai/ui-library';
import type { Store } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import { $isDebugging } from 'app/store/nanostores/isDebugging';
import type { RootState } from 'app/store/store';
import { setStageEventHandlers } from 'features/controlLayers/konva/events';
import { KonvaNodeManager } from 'features/controlLayers/konva/nodeManager';
import { getArrangeEntities } from 'features/controlLayers/konva/renderers/arrange';
import { createBackgroundLayer, getRenderBackground } from 'features/controlLayers/konva/renderers/background';
import { updateBboxes } from 'features/controlLayers/konva/renderers/bbox';
import { getRenderControlAdapters } from 'features/controlLayers/konva/renderers/controlAdapters';
import { getRenderInpaintMask } from 'features/controlLayers/konva/renderers/inpaintMask';
import { getRenderLayers } from 'features/controlLayers/konva/renderers/layers';
import {
  createBboxNodes,
  createDocumentOverlay,
  createPreviewLayer,
  createToolPreviewNodes,
  getRenderBbox,
  getRenderDocumentOverlay,
  getRenderToolPreview,
} from 'features/controlLayers/konva/renderers/preview';
import { getRenderRegions } from 'features/controlLayers/konva/renderers/regions';
import { getFitDocumentToStage, getFitStageToContainer } from 'features/controlLayers/konva/renderers/stage';
import {
  $stageAttrs,
  bboxChanged,
  brushWidthChanged,
  caBboxChanged,
  caTranslated,
  eraserWidthChanged,
  imBboxChanged,
  imBrushLineAdded,
  imEraserLineAdded,
  imImageCacheChanged,
  imLinePointAdded,
  imTranslated,
  layerBboxChanged,
  layerBrushLineAdded,
  layerEraserLineAdded,
  layerImageCacheChanged,
  layerLinePointAdded,
  layerRectAdded,
  layerTranslated,
  rgBboxChanged,
  rgBrushLineAdded,
  rgEraserLineAdded,
  rgImageCacheChanged,
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
import { atom } from 'nanostores';
import type { RgbaColor } from 'react-colorful';
import type { ImageDTO } from 'services/api/types';

export const $nodeManager = atom<KonvaNodeManager | null>(null);

/**
 * Initializes the canvas renderer. It subscribes to the redux store and listens for changes directly, bypassing the
 * react rendering cycle entirely, improving canvas performance.
 * @param store The redux store
 * @param stage The konva stage
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
    } else if (entityType === 'inpaint_mask') {
      dispatch(imTranslated(arg));
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
    } else if (entityType === 'inpaint_mask') {
      dispatch(imBboxChanged(arg));
    }
  };
  const onBrushLineAdded = (arg: BrushLineAddedArg, entityType: CanvasEntity['type']) => {
    logIfDebugging('Brush line added');
    if (entityType === 'layer') {
      dispatch(layerBrushLineAdded(arg));
    } else if (entityType === 'regional_guidance') {
      dispatch(rgBrushLineAdded(arg));
    } else if (entityType === 'inpaint_mask') {
      dispatch(imBrushLineAdded(arg));
    }
  };
  const onEraserLineAdded = (arg: EraserLineAddedArg, entityType: CanvasEntity['type']) => {
    logIfDebugging('Eraser line added');
    if (entityType === 'layer') {
      dispatch(layerEraserLineAdded(arg));
    } else if (entityType === 'regional_guidance') {
      dispatch(rgEraserLineAdded(arg));
    } else if (entityType === 'inpaint_mask') {
      dispatch(imEraserLineAdded(arg));
    }
  };
  const onPointAddedToLine = (arg: PointAddedToLineArg, entityType: CanvasEntity['type']) => {
    logIfDebugging('Point added to line');
    if (entityType === 'layer') {
      dispatch(layerLinePointAdded(arg));
    } else if (entityType === 'regional_guidance') {
      dispatch(rgLinePointAdded(arg));
    } else if (entityType === 'inpaint_mask') {
      dispatch(imLinePointAdded(arg));
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
  const onRegionMaskImageCached = (id: string, imageDTO: ImageDTO) => {
    logIfDebugging('Region mask image cached');
    dispatch(rgImageCacheChanged({ id, imageDTO }));
  };
  const onInpaintMaskImageCached = (imageDTO: ImageDTO) => {
    logIfDebugging('Inpaint mask image cached');
    dispatch(imImageCacheChanged({ imageDTO }));
  };
  const onLayerImageCached = (imageDTO: ImageDTO) => {
    logIfDebugging('Layer image cached');
    dispatch(layerImageCacheChanged({ imageDTO }));
  };

  const setTool = (tool: Tool) => {
    logIfDebugging('Tool selection changed');
    dispatch(toolChanged(tool));
  };
  const setToolBuffer = (toolBuffer: Tool | null) => {
    logIfDebugging('Tool buffer changed');
    dispatch(toolBufferChanged(toolBuffer));
  };

  const selectSelectedEntity = (canvasV2: CanvasV2State): CanvasEntity | null => {
    const identifier = canvasV2.selectedEntityIdentifier;
    let selectedEntity: CanvasEntity | null = null;
    if (!identifier) {
      selectedEntity = null;
    } else if (identifier.type === 'layer') {
      selectedEntity = canvasV2.layers.entities.find((i) => i.id === identifier.id) ?? null;
    } else if (identifier.type === 'control_adapter') {
      selectedEntity = canvasV2.controlAdapters.entities.find((i) => i.id === identifier.id) ?? null;
    } else if (identifier.type === 'ip_adapter') {
      selectedEntity = canvasV2.ipAdapters.entities.find((i) => i.id === identifier.id) ?? null;
    } else if (identifier.type === 'regional_guidance') {
      selectedEntity = canvasV2.regions.entities.find((i) => i.id === identifier.id) ?? null;
    } else if (identifier.type === 'inpaint_mask') {
      selectedEntity = canvasV2.inpaintMask;
    } else {
      selectedEntity = null;
    }
    logIfDebugging('Selected entity changed');
    return selectedEntity;
  };

  const selectCurrentFill = (canvasV2: CanvasV2State, selectedEntity: CanvasEntity | null) => {
    let currentFill: RgbaColor = canvasV2.tool.fill;
    if (selectedEntity) {
      if (selectedEntity.type === 'regional_guidance') {
        currentFill = { ...selectedEntity.fill, a: canvasV2.settings.maskOpacity };
      } else if (selectedEntity.type === 'inpaint_mask') {
        currentFill = { ...canvasV2.inpaintMask.fill, a: canvasV2.settings.maskOpacity };
      }
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
  let canvasV2 = getState().canvasV2;
  let prevSelectedEntity: CanvasEntity | null = selectSelectedEntity(prevCanvasV2);
  let prevCurrentFill: RgbaColor = selectCurrentFill(prevCanvasV2, prevSelectedEntity);
  const getSelectedEntity = () => prevSelectedEntity;
  const getCurrentFill = () => prevCurrentFill;
  const getBbox = () => canvasV2.bbox;
  const getDocument = () => canvasV2.document;
  const getToolState = () => canvasV2.tool;
  const getSettings = () => canvasV2.settings;
  const getRegionsState = () => canvasV2.regions;
  const getLayersState = () => canvasV2.layers;
  const getControlAdaptersState = () => canvasV2.controlAdapters;
  const getInpaintMaskState = () => canvasV2.inpaintMask;
  const getMaskOpacity = () => canvasV2.settings.maskOpacity;

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

  const manager = new KonvaNodeManager(stage, container);
  $nodeManager.set(manager);

  manager.background = { layer: createBackgroundLayer() };
  manager.stage.add(manager.background.layer);
  manager.preview = {
    layer: createPreviewLayer(),
    bbox: createBboxNodes(stage, getBbox, onBboxTransformed, $shift.get, $ctrl.get, $meta.get, $alt.get),
    tool: createToolPreviewNodes(),
    documentOverlay: createDocumentOverlay(),
  };
  manager.preview.layer.add(manager.preview.bbox.group);
  manager.preview.layer.add(manager.preview.tool.group);
  manager.preview.layer.add(manager.preview.documentOverlay.group);
  manager.stage.add(manager.preview.layer);
  manager.stateApi = {
    // Read-only state
    getToolState,
    getSelectedEntity,
    getBbox,
    getSettings,
    getCurrentFill,
    getAltKey: $alt.get,
    getCtrlKey: $ctrl.get,
    getMetaKey: $meta.get,
    getShiftKey: $shift.get,
    getControlAdaptersState,
    getDocument,
    getLayersState,
    getRegionsState,
    getMaskOpacity,
    getInpaintMaskState,

    // Read-write state
    setTool,
    setToolBuffer,
    getIsDrawing,
    setIsDrawing,
    getIsMouseDown,
    setIsMouseDown,
    getLastAddedPoint,
    setLastAddedPoint,
    getLastCursorPos,
    setLastCursorPos,
    getLastMouseDownPos,
    setLastMouseDownPos,
    getSpaceKey,
    setSpaceKey,
    setStageAttrs: $stageAttrs.set,

    // Callbacks
    onBrushLineAdded,
    onEraserLineAdded,
    onPointAddedToLine,
    onRectShapeAdded,
    onBrushWidthChanged,
    onEraserWidthChanged,
    onPosChanged,
    onBboxTransformed,
    onRegionMaskImageCached,
    onInpaintMaskImageCached,
    onLayerImageCached,
  };

  const cleanupListeners = setStageEventHandlers(manager);

  // Calculating bounding boxes is expensive, must be debounced to not block the UI thread during a user interaction.
  // TODO(psyche): Figure out how to do this in a worker. Probably means running the renderer in a worker and sending
  // the entire state over when needed.
  const debouncedUpdateBboxes = debounce(updateBboxes, 300);

  manager.konvaApi = {
    renderRegions: getRenderRegions(manager),
    renderLayers: getRenderLayers(manager),
    renderControlAdapters: getRenderControlAdapters(manager),
    renderInpaintMask: getRenderInpaintMask(manager),
    renderBbox: getRenderBbox(manager),
    renderToolPreview: getRenderToolPreview(manager),
    renderDocumentOverlay: getRenderDocumentOverlay(manager),
    renderBackground: getRenderBackground(manager),
    arrangeEntities: getArrangeEntities(manager),
    fitDocumentToStage: getFitDocumentToStage(manager),
    fitStageToContainer: getFitStageToContainer(manager),
  };

  const renderCanvas = () => {
    canvasV2 = store.getState().canvasV2;

    if (prevCanvasV2 === canvasV2 && !isFirstRender) {
      logIfDebugging('No changes detected, skipping render');
      return;
    }

    const selectedEntity = selectSelectedEntity(canvasV2);
    const currentFill = selectCurrentFill(canvasV2, selectedEntity);

    if (
      isFirstRender ||
      canvasV2.layers.entities !== prevCanvasV2.layers.entities ||
      canvasV2.tool.selected !== prevCanvasV2.tool.selected
    ) {
      logIfDebugging('Rendering layers');
      manager.konvaApi.renderLayers();
    }

    if (
      isFirstRender ||
      canvasV2.regions.entities !== prevCanvasV2.regions.entities ||
      canvasV2.settings.maskOpacity !== prevCanvasV2.settings.maskOpacity ||
      canvasV2.tool.selected !== prevCanvasV2.tool.selected
    ) {
      logIfDebugging('Rendering regions');
      manager.konvaApi.renderRegions();
    }

    if (
      isFirstRender ||
      canvasV2.inpaintMask !== prevCanvasV2.inpaintMask ||
      canvasV2.settings.maskOpacity !== prevCanvasV2.settings.maskOpacity ||
      canvasV2.tool.selected !== prevCanvasV2.tool.selected
    ) {
      logIfDebugging('Rendering inpaint mask');
      manager.konvaApi.renderInpaintMask();
    }

    if (isFirstRender || canvasV2.controlAdapters.entities !== prevCanvasV2.controlAdapters.entities) {
      logIfDebugging('Rendering control adapters');
      manager.konvaApi.renderControlAdapters();
    }

    if (isFirstRender || canvasV2.document !== prevCanvasV2.document) {
      logIfDebugging('Rendering document bounds overlay');
      manager.konvaApi.renderDocumentOverlay();
    }

    if (isFirstRender || canvasV2.bbox !== prevCanvasV2.bbox || canvasV2.tool.selected !== prevCanvasV2.tool.selected) {
      logIfDebugging('Rendering generation bbox');
      manager.konvaApi.renderBbox();
    }

    if (
      isFirstRender ||
      canvasV2.layers !== prevCanvasV2.layers ||
      canvasV2.controlAdapters !== prevCanvasV2.controlAdapters ||
      canvasV2.regions !== prevCanvasV2.regions
    ) {
      // logIfDebugging('Updating entity bboxes');
      // debouncedUpdateBboxes(stage, canvasV2.layers, canvasV2.controlAdapters, canvasV2.regions, onBboxChanged);
    }

    if (
      isFirstRender ||
      canvasV2.layers.entities !== prevCanvasV2.layers.entities ||
      canvasV2.controlAdapters.entities !== prevCanvasV2.controlAdapters.entities ||
      canvasV2.regions.entities !== prevCanvasV2.regions.entities
    ) {
      logIfDebugging('Arranging entities');
      manager.konvaApi.arrangeEntities();
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
  const resizeObserver = new ResizeObserver(manager.konvaApi.fitStageToContainer);
  resizeObserver.observe(container);
  manager.konvaApi.fitStageToContainer();

  const unsubscribeRenderer = subscribe(renderCanvas);

  logIfDebugging('First render of konva stage');
  // On first render, the document should be fit to the stage.
  manager.konvaApi.fitDocumentToStage();
  manager.konvaApi.renderToolPreview();
  renderCanvas();

  return () => {
    logIfDebugging('Cleaning up konva renderer');
    unsubscribeRenderer();
    cleanupListeners();
    resizeObserver.disconnect();
  };
};
