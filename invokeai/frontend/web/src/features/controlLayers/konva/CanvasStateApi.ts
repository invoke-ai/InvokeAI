import { $alt, $ctrl, $meta, $shift } from '@invoke-ai/ui-library';
import type { Store } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import type { RootState } from 'app/store/store';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import {
  $isDrawing,
  $isMouseDown,
  $lastAddedPoint,
  $lastCursorPos,
  $lastMouseDownPos,
  $lastProgressEvent,
  $shouldShowStagedImage,
  $spaceKey,
  $stageAttrs,
  bboxChanged,
  brushWidthChanged,
  entityBrushLineAdded,
  entityEraserLineAdded,
  entityMoved,
  entityRasterized,
  entityRectAdded,
  entityReset,
  entitySelected,
  eraserWidthChanged,
  imImageCacheChanged,
  layerImageCacheChanged,
  rgImageCacheChanged,
  toolBufferChanged,
  toolChanged,
} from 'features/controlLayers/store/canvasV2Slice';
import type {
  EntityBrushLineAddedPayload,
  EntityEraserLineAddedPayload,
  EntityIdentifierPayload,
  EntityMovedPayload,
  EntityRasterizedPayload,
  EntityRectAddedPayload,
  Rect,
  Tool,
} from 'features/controlLayers/store/types';
import type { ImageDTO } from 'services/api/types';

const log = logger('canvas');

export class CanvasStateApi {
  _store: Store<RootState>;
  manager: CanvasManager;

  constructor(store: Store<RootState>, manager: CanvasManager) {
    this._store = store;
    this.manager = manager;
  }

  // Reminder - use arrow functions to avoid binding issues
  getState = () => {
    return this._store.getState().canvasV2;
  };
  resetEntity = (arg: EntityIdentifierPayload) => {
    log.trace(arg, 'Resetting entity');
    this._store.dispatch(entityReset(arg));
  };
  setEntityPosition = (arg: EntityMovedPayload) => {
    log.trace(arg, 'Setting entity position');
    this._store.dispatch(entityMoved(arg));
  };
  addBrushLine = (arg: EntityBrushLineAddedPayload) => {
    log.trace(arg, 'Adding brush line');
    this._store.dispatch(entityBrushLineAdded(arg));
  };
  addEraserLine = (arg: EntityEraserLineAddedPayload) => {
    log.trace(arg, 'Adding eraser line');
    this._store.dispatch(entityEraserLineAdded(arg));
  };
  addRect = (arg: EntityRectAddedPayload) => {
    log.trace(arg, 'Adding rect');
    this._store.dispatch(entityRectAdded(arg));
  };
  rasterizeEntity = (arg: EntityRasterizedPayload) => {
    log.trace(arg, 'Rasterizing entity');
    this._store.dispatch(entityRasterized(arg));
  };
  setSelectedEntity = (arg: EntityIdentifierPayload) => {
    log.trace({ arg }, 'Setting selected entity');
    this._store.dispatch(entitySelected(arg));
  };
  setGenerationBbox = (bbox: Rect) => {
    log.trace({ bbox }, 'Setting generation bbox');
    this._store.dispatch(bboxChanged(bbox));
  };
  setBrushWidth = (width: number) => {
    log.trace({ width }, 'Setting brush width');
    this._store.dispatch(brushWidthChanged(width));
  };
  setEraserWidth = (width: number) => {
    log.trace({ width }, 'Setting eraser width');
    this._store.dispatch(eraserWidthChanged(width));
  };
  setRegionMaskImageCache = (id: string, imageDTO: ImageDTO) => {
    log.trace({ id, imageDTO }, 'Setting region mask image cache');
    this._store.dispatch(rgImageCacheChanged({ id, imageDTO }));
  };
  setInpaintMaskImageCache = (imageDTO: ImageDTO) => {
    log.trace({ imageDTO }, 'Setting inpaint mask image cache');
    this._store.dispatch(imImageCacheChanged({ imageDTO }));
  };
  setLayerImageCache = (imageDTO: ImageDTO) => {
    log.trace({ imageDTO }, 'Setting layer image cache');
    this._store.dispatch(layerImageCacheChanged({ imageDTO }));
  };
  setTool = (tool: Tool) => {
    log.trace({ tool }, 'Setting tool');
    this._store.dispatch(toolChanged(tool));
  };
  setToolBuffer = (toolBuffer: Tool | null) => {
    log.trace({ toolBuffer }, 'Setting tool buffer');
    this._store.dispatch(toolBufferChanged(toolBuffer));
  };

  getBbox = () => {
    return this.getState().bbox;
  };

  getToolState = () => {
    return this.getState().tool;
  };
  getSettings = () => {
    return this.getState().settings;
  };
  getRegionsState = () => {
    return this.getState().regions;
  };
  getLayersState = () => {
    return this.getState().layers;
  };
  getControlAdaptersState = () => {
    return this.getState().controlAdapters;
  };
  getInpaintMaskState = () => {
    return this.getState().inpaintMask;
  };
  getMaskOpacity = () => {
    return this.getState().settings.maskOpacity;
  };
  getSession = () => {
    return this.getState().session;
  };
  getIsSelected = (id: string) => {
    return this.getState().selectedEntityIdentifier?.id === id;
  };
  getLogLevel = () => {
    return this._store.getState().system.consoleLogLevel;
  };

  // Read-write state, ephemeral interaction state
  $isDrawing = $isDrawing;
  $isMouseDown = $isMouseDown;
  $lastAddedPoint = $lastAddedPoint;
  $lastMouseDownPos = $lastMouseDownPos;
  $lastCursorPos = $lastCursorPos;
  $lastProgressEvent = $lastProgressEvent;
  $spaceKey = $spaceKey;
  $altKey = $alt;
  $ctrlKey = $ctrl;
  $metaKey = $meta;
  $shiftKey = $shift;
  $shouldShowStagedImage = $shouldShowStagedImage;
  $stageAttrs = $stageAttrs;
}
