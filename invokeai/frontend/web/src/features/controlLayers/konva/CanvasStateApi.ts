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
  caTranslated,
  entitySelected,
  eraserWidthChanged,
  imBrushLineAdded,
  imEraserLineAdded,
  imImageCacheChanged,
  imMoved,
  imRectAdded,
  inpaintMaskRasterized,
  layerBrushLineAdded,
  layerEraserLineAdded,
  layerImageCacheChanged,
  layerRasterized,
  layerRectAdded,
  layerReset,
  layerTranslated,
  rgBrushLineAdded,
  rgEraserLineAdded,
  rgImageCacheChanged,
  rgMoved,
  rgRasterized,
  rgRectAdded,
  toolBufferChanged,
  toolChanged,
} from 'features/controlLayers/store/canvasV2Slice';
import type {
  CanvasBrushLineState,
  CanvasEntityIdentifier,
  CanvasEntityState,
  CanvasEraserLineState,
  CanvasRectState,
  EntityRasterizedArg,
  PositionChangedArg,
  Rect,
  Tool,
} from 'features/controlLayers/store/types';
import type { ImageDTO } from 'services/api/types';
import { assert } from 'tsafe';

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
  resetEntity = (arg: { id: string }, entityType: CanvasEntityState['type']) => {
    log.trace({ arg, entityType }, 'Resetting entity');
    if (entityType === 'layer') {
      this._store.dispatch(layerReset(arg));
    }
  };
  setEntityPosition = (arg: PositionChangedArg, entityType: CanvasEntityState['type']) => {
    log.trace({ arg, entityType }, 'Setting entity position');
    if (entityType === 'layer') {
      this._store.dispatch(layerTranslated(arg));
    } else if (entityType === 'regional_guidance') {
      this._store.dispatch(rgMoved(arg));
    } else if (entityType === 'inpaint_mask') {
      this._store.dispatch(imMoved(arg));
    } else if (entityType === 'control_adapter') {
      this._store.dispatch(caTranslated(arg));
    }
  };
  addBrushLine = (arg: { id: string; brushLine: CanvasBrushLineState }, entityType: CanvasEntityState['type']) => {
    log.trace({ arg, entityType }, 'Adding brush line');
    if (entityType === 'layer') {
      this._store.dispatch(layerBrushLineAdded(arg));
    } else if (entityType === 'regional_guidance') {
      this._store.dispatch(rgBrushLineAdded(arg));
    } else if (entityType === 'inpaint_mask') {
      this._store.dispatch(imBrushLineAdded(arg));
    }
  };
  addEraserLine = (arg: { id: string; eraserLine: CanvasEraserLineState }, entityType: CanvasEntityState['type']) => {
    log.trace({ arg, entityType }, 'Adding eraser line');
    if (entityType === 'layer') {
      this._store.dispatch(layerEraserLineAdded(arg));
    } else if (entityType === 'regional_guidance') {
      this._store.dispatch(rgEraserLineAdded(arg));
    } else if (entityType === 'inpaint_mask') {
      this._store.dispatch(imEraserLineAdded(arg));
    }
  };
  addRect = (arg: { id: string; rect: CanvasRectState }, entityType: CanvasEntityState['type']) => {
    log.trace({ arg, entityType }, 'Adding rect');
    if (entityType === 'layer') {
      this._store.dispatch(layerRectAdded(arg));
    } else if (entityType === 'regional_guidance') {
      this._store.dispatch(rgRectAdded(arg));
    } else if (entityType === 'inpaint_mask') {
      this._store.dispatch(imRectAdded(arg));
    }
  };
  rasterizeEntity = (arg: EntityRasterizedArg, entityType: CanvasEntityState['type']) => {
    log.trace({ arg, entityType }, 'Rasterizing entity');
    if (entityType === 'layer') {
      this._store.dispatch(layerRasterized(arg));
    } else if (entityType === 'inpaint_mask') {
      this._store.dispatch(inpaintMaskRasterized(arg));
    } else if (entityType === 'regional_guidance') {
      this._store.dispatch(rgRasterized(arg));
    } else {
      assert(false, 'Rasterizing not supported for this entity type');
    }
  };
  setSelectedEntity = (arg: CanvasEntityIdentifier) => {
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
