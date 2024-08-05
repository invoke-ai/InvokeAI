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
  caBboxChanged,
  caScaled,
  caTranslated,
  entitySelected,
  eraserWidthChanged,
  imBboxChanged,
  imBrushLineAdded,
  imEraserLineAdded,
  imImageCacheChanged,
  imRectShapeAdded,
  imScaled,
  imTranslated,
  layerBboxChanged,
  layerBrushLineAdded,
  layerEraserLineAdded,
  layerImageCacheChanged,
  layerRectShapeAdded,
  layerReset,
  layerTranslated,
  rgBboxChanged,
  rgBrushLineAdded,
  rgEraserLineAdded,
  rgImageCacheChanged,
  rgRectShapeAdded,
  rgScaled,
  rgTranslated,
  toolBufferChanged,
  toolChanged,
} from 'features/controlLayers/store/canvasV2Slice';
import type {
  BboxChangedArg,
  CanvasBrushLineState,
  CanvasEntity,
  CanvasEraserLineState,
  CanvasRectState,
  PositionChangedArg,
  ScaleChangedArg,
  Tool,
} from 'features/controlLayers/store/types';
import type { IRect } from 'konva/lib/types';
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
  onEntityReset = (arg: { id: string }, entityType: CanvasEntity['type']) => {
    log.debug('onEntityReset');
    if (entityType === 'layer') {
      this._store.dispatch(layerReset(arg));
    }
  };
  onPosChanged = (arg: PositionChangedArg, entityType: CanvasEntity['type']) => {
    log.debug('onPosChanged');
    if (entityType === 'layer') {
      this._store.dispatch(layerTranslated(arg));
    } else if (entityType === 'regional_guidance') {
      this._store.dispatch(rgTranslated(arg));
    } else if (entityType === 'inpaint_mask') {
      this._store.dispatch(imTranslated(arg));
    } else if (entityType === 'control_adapter') {
      this._store.dispatch(caTranslated(arg));
    }
  };
  onScaleChanged = (arg: ScaleChangedArg, entityType: CanvasEntity['type']) => {
    log.debug('onScaleChanged');
    if (entityType === 'inpaint_mask') {
      this._store.dispatch(imScaled(arg));
    } else if (entityType === 'regional_guidance') {
      this._store.dispatch(rgScaled(arg));
    } else if (entityType === 'control_adapter') {
      this._store.dispatch(caScaled(arg));
    }
  };
  onBboxChanged = (arg: BboxChangedArg, entityType: CanvasEntity['type']) => {
    log.debug('Entity bbox changed');
    if (entityType === 'layer') {
      this._store.dispatch(layerBboxChanged(arg));
    } else if (entityType === 'control_adapter') {
      this._store.dispatch(caBboxChanged(arg));
    } else if (entityType === 'regional_guidance') {
      this._store.dispatch(rgBboxChanged(arg));
    } else if (entityType === 'inpaint_mask') {
      this._store.dispatch(imBboxChanged(arg));
    }
  };
  onBrushLineAdded = (arg: { id: string; brushLine: CanvasBrushLineState }, entityType: CanvasEntity['type']) => {
    log.debug('Brush line added');
    if (entityType === 'layer') {
      this._store.dispatch(layerBrushLineAdded(arg));
    } else if (entityType === 'regional_guidance') {
      this._store.dispatch(rgBrushLineAdded(arg));
    } else if (entityType === 'inpaint_mask') {
      this._store.dispatch(imBrushLineAdded(arg));
    }
  };
  onEraserLineAdded = (arg: { id: string; eraserLine: CanvasEraserLineState }, entityType: CanvasEntity['type']) => {
    log.debug('Eraser line added');
    if (entityType === 'layer') {
      this._store.dispatch(layerEraserLineAdded(arg));
    } else if (entityType === 'regional_guidance') {
      this._store.dispatch(rgEraserLineAdded(arg));
    } else if (entityType === 'inpaint_mask') {
      this._store.dispatch(imEraserLineAdded(arg));
    }
  };
  onRectShapeAdded = (arg: { id: string; rectShape: CanvasRectState }, entityType: CanvasEntity['type']) => {
    log.debug('Rect shape added');
    if (entityType === 'layer') {
      this._store.dispatch(layerRectShapeAdded(arg));
    } else if (entityType === 'regional_guidance') {
      this._store.dispatch(rgRectShapeAdded(arg));
    } else if (entityType === 'inpaint_mask') {
      this._store.dispatch(imRectShapeAdded(arg));
    }
  };
  onEntitySelected = (arg: { id: string; type: CanvasEntity['type'] }) => {
    log.debug('Entity selected');
    this._store.dispatch(entitySelected(arg));
  };
  onBboxTransformed = (bbox: IRect) => {
    log.debug('Generation bbox transformed');
    this._store.dispatch(bboxChanged(bbox));
  };
  onBrushWidthChanged = (width: number) => {
    log.debug('Brush width changed');
    this._store.dispatch(brushWidthChanged(width));
  };
  onEraserWidthChanged = (width: number) => {
    log.debug('Eraser width changed');
    this._store.dispatch(eraserWidthChanged(width));
  };
  onRegionMaskImageCached = (id: string, imageDTO: ImageDTO) => {
    log.debug('Region mask image cached');
    this._store.dispatch(rgImageCacheChanged({ id, imageDTO }));
  };
  onInpaintMaskImageCached = (imageDTO: ImageDTO) => {
    log.debug('Inpaint mask image cached');
    this._store.dispatch(imImageCacheChanged({ imageDTO }));
  };
  onLayerImageCached = (imageDTO: ImageDTO) => {
    log.debug('Layer image cached');
    this._store.dispatch(layerImageCacheChanged({ imageDTO }));
  };
  setTool = (tool: Tool) => {
    log.debug('Tool selection changed');
    this._store.dispatch(toolChanged(tool));
  };
  setToolBuffer = (toolBuffer: Tool | null) => {
    log.debug('Tool buffer changed');
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
  getInitialImageState = () => {
    return this.getState().initialImage;
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
