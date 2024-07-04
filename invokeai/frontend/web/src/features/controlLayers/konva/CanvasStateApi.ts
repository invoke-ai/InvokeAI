import { $alt, $ctrl, $meta, $shift } from '@invoke-ai/ui-library';
import type { Store } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import type { RootState } from 'app/store/store';
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
  caTranslated,
  eraserWidthChanged,
  imBboxChanged,
  imBrushLineAdded,
  imEraserLineAdded,
  imImageCacheChanged,
  imLinePointAdded,
  imRectAdded,
  imScaled,
  imTranslated,
  layerBboxChanged,
  layerBrushLineAdded,
  layerEraserLineAdded,
  layerImageCacheChanged,
  layerLinePointAdded,
  layerRectAdded,
  layerScaled,
  layerTranslated,
  rgBboxChanged,
  rgBrushLineAdded,
  rgEraserLineAdded,
  rgImageCacheChanged,
  rgLinePointAdded,
  rgRectAdded,
  rgScaled,
  rgTranslated,
  toolBufferChanged,
  toolChanged,
} from 'features/controlLayers/store/canvasV2Slice';
import type {
  BboxChangedArg,
  BrushLineAddedArg,
  CanvasEntity,
  EraserLineAddedArg,
  PointAddedToLineArg,
  PosChangedArg,
  RectShapeAddedArg,
  ScaleChangedArg,
  Tool,
} from 'features/controlLayers/store/types';
import type { IRect } from 'konva/lib/types';
import type { RgbaColor } from 'react-colorful';
import type { ImageDTO } from 'services/api/types';

const log = logger('canvas');

export class CanvasStateApi {
  private store: Store<RootState>;

  constructor(store: Store<RootState>) {
    this.store = store;
  }

  // Reminder - use arrow functions to avoid binding issues
  getState = () => {
    return this.store.getState().canvasV2;
  };

  onPosChanged = (arg: PosChangedArg, entityType: CanvasEntity['type']) => {
    log.debug('onPosChanged');
    if (entityType === 'layer') {
      this.store.dispatch(layerTranslated(arg));
    } else if (entityType === 'control_adapter') {
      this.store.dispatch(caTranslated(arg));
    } else if (entityType === 'regional_guidance') {
      this.store.dispatch(rgTranslated(arg));
    } else if (entityType === 'inpaint_mask') {
      this.store.dispatch(imTranslated(arg));
    }
  };
  onScaleChanged = (arg: ScaleChangedArg, entityType: CanvasEntity['type']) => {
    log.debug('onScaleChanged');
    if (entityType === 'layer') {
      this.store.dispatch(layerScaled(arg));
    } else if (entityType === 'inpaint_mask') {
      this.store.dispatch(imScaled(arg));
    } else if (entityType === 'regional_guidance') {
      this.store.dispatch(rgScaled(arg));
    }
  };
  onBboxChanged = (arg: BboxChangedArg, entityType: CanvasEntity['type']) => {
    log.debug('Entity bbox changed');
    if (entityType === 'layer') {
      this.store.dispatch(layerBboxChanged(arg));
    } else if (entityType === 'control_adapter') {
      this.store.dispatch(caBboxChanged(arg));
    } else if (entityType === 'regional_guidance') {
      this.store.dispatch(rgBboxChanged(arg));
    } else if (entityType === 'inpaint_mask') {
      this.store.dispatch(imBboxChanged(arg));
    }
  };
  onBrushLineAdded = (arg: BrushLineAddedArg, entityType: CanvasEntity['type']) => {
    log.debug('Brush line added');
    if (entityType === 'layer') {
      this.store.dispatch(layerBrushLineAdded(arg));
    } else if (entityType === 'regional_guidance') {
      this.store.dispatch(rgBrushLineAdded(arg));
    } else if (entityType === 'inpaint_mask') {
      this.store.dispatch(imBrushLineAdded(arg));
    }
  };
  onEraserLineAdded = (arg: EraserLineAddedArg, entityType: CanvasEntity['type']) => {
    log.debug('Eraser line added');
    if (entityType === 'layer') {
      this.store.dispatch(layerEraserLineAdded(arg));
    } else if (entityType === 'regional_guidance') {
      this.store.dispatch(rgEraserLineAdded(arg));
    } else if (entityType === 'inpaint_mask') {
      this.store.dispatch(imEraserLineAdded(arg));
    }
  };
  onPointAddedToLine = (arg: PointAddedToLineArg, entityType: CanvasEntity['type']) => {
    log.debug('Point added to line');
    if (entityType === 'layer') {
      this.store.dispatch(layerLinePointAdded(arg));
    } else if (entityType === 'regional_guidance') {
      this.store.dispatch(rgLinePointAdded(arg));
    } else if (entityType === 'inpaint_mask') {
      this.store.dispatch(imLinePointAdded(arg));
    }
  };
  onRectShapeAdded = (arg: RectShapeAddedArg, entityType: CanvasEntity['type']) => {
    log.debug('Rect shape added');
    if (entityType === 'layer') {
      this.store.dispatch(layerRectAdded(arg));
    } else if (entityType === 'regional_guidance') {
      this.store.dispatch(rgRectAdded(arg));
    } else if (entityType === 'inpaint_mask') {
      this.store.dispatch(imRectAdded(arg));
    }
  };
  onBboxTransformed = (bbox: IRect) => {
    log.debug('Generation bbox transformed');
    this.store.dispatch(bboxChanged(bbox));
  };
  onBrushWidthChanged = (width: number) => {
    log.debug('Brush width changed');
    this.store.dispatch(brushWidthChanged(width));
  };
  onEraserWidthChanged = (width: number) => {
    log.debug('Eraser width changed');
    this.store.dispatch(eraserWidthChanged(width));
  };
  onRegionMaskImageCached = (id: string, imageDTO: ImageDTO) => {
    log.debug('Region mask image cached');
    this.store.dispatch(rgImageCacheChanged({ id, imageDTO }));
  };
  onInpaintMaskImageCached = (imageDTO: ImageDTO) => {
    log.debug('Inpaint mask image cached');
    this.store.dispatch(imImageCacheChanged({ imageDTO }));
  };
  onLayerImageCached = (imageDTO: ImageDTO) => {
    log.debug('Layer image cached');
    this.store.dispatch(layerImageCacheChanged({ imageDTO }));
  };
  setTool = (tool: Tool) => {
    log.debug('Tool selection changed');
    this.store.dispatch(toolChanged(tool));
  };
  setToolBuffer = (toolBuffer: Tool | null) => {
    log.debug('Tool buffer changed');
    this.store.dispatch(toolBufferChanged(toolBuffer));
  };

  getSelectedEntity = (): CanvasEntity | null => {
    const state = this.getState();
    const identifier = state.selectedEntityIdentifier;
    let selectedEntity: CanvasEntity | null = null;
    if (!identifier) {
      selectedEntity = null;
    } else if (identifier.type === 'layer') {
      selectedEntity = state.layers.entities.find((i) => i.id === identifier.id) ?? null;
    } else if (identifier.type === 'control_adapter') {
      selectedEntity = state.controlAdapters.entities.find((i) => i.id === identifier.id) ?? null;
    } else if (identifier.type === 'ip_adapter') {
      selectedEntity = state.ipAdapters.entities.find((i) => i.id === identifier.id) ?? null;
    } else if (identifier.type === 'regional_guidance') {
      selectedEntity = state.regions.entities.find((i) => i.id === identifier.id) ?? null;
    } else if (identifier.type === 'inpaint_mask') {
      selectedEntity = state.inpaintMask;
    } else {
      selectedEntity = null;
    }
    return selectedEntity;
  };

  getCurrentFill = () => {
    const state = this.getState();
    const selectedEntity = this.getSelectedEntity();
    let currentFill: RgbaColor = state.tool.fill;
    if (selectedEntity) {
      if (selectedEntity.type === 'regional_guidance') {
        currentFill = { ...selectedEntity.fill, a: state.settings.maskOpacity };
      } else if (selectedEntity.type === 'inpaint_mask') {
        currentFill = { ...state.inpaintMask.fill, a: state.settings.maskOpacity };
      }
    } else {
      currentFill = state.tool.fill;
    }
    return currentFill;
  };
  getBbox = () => {
    return this.getState().bbox;
  };
  getDocument = () => {
    return this.getState().document;
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
  getStagingAreaState = () => {
    return this.getState().stagingArea;
  };
  getIsSelected = (id: string) => {
    return this.getSelectedEntity()?.id === id;
  };

  // Read-only state, derived from nanostores
  resetLastProgressEvent = () => {
    $lastProgressEvent.set(null);
  };

  // Read-write state, ephemeral interaction state
  getIsDrawing = $isDrawing.get;
  setIsDrawing = $isDrawing.set;

  getIsMouseDown = $isMouseDown.get;
  setIsMouseDown = $isMouseDown.set;

  getLastAddedPoint = $lastAddedPoint.get;
  setLastAddedPoint = $lastAddedPoint.set;

  getLastMouseDownPos = $lastMouseDownPos.get;
  setLastMouseDownPos = $lastMouseDownPos.set;

  getLastCursorPos = $lastCursorPos.get;
  setLastCursorPos = $lastCursorPos.set;

  getSpaceKey = $spaceKey.get;
  setSpaceKey = $spaceKey.set;

  getLastProgressEvent = $lastProgressEvent.get;
  setLastProgressEvent = $lastProgressEvent.set;

  getAltKey = $alt.get;
  getCtrlKey = $ctrl.get;
  getMetaKey = $meta.get;
  getShiftKey = $shift.get;

  getShouldShowStagedImage = $shouldShowStagedImage.get;
  setStageAttrs = $stageAttrs.set;
}
