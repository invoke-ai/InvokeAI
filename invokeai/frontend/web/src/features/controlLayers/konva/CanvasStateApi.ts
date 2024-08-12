import { $alt, $ctrl, $meta, $shift } from '@invoke-ai/ui-library';
import type { Store } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import type { RootState } from 'app/store/store';
import type { CanvasLayerAdapter } from 'features/controlLayers/konva/CanvasLayerAdapter';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { CanvasMaskAdapter } from 'features/controlLayers/konva/CanvasMaskAdapter';
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
  CanvasEntityIdentifier,
  CanvasInpaintMaskState,
  CanvasLayerState,
  CanvasRegionalGuidanceState,
  CanvasV2State,
  EntityBrushLineAddedPayload,
  EntityEraserLineAddedPayload,
  EntityIdentifierPayload,
  EntityMovedPayload,
  EntityRasterizedPayload,
  EntityRectAddedPayload,
  Rect,
  RgbaColor,
  Tool,
} from 'features/controlLayers/store/types';
import { RGBA_RED } from 'features/controlLayers/store/types';
import type { WritableAtom } from 'nanostores';
import { atom } from 'nanostores';
import type { ImageDTO } from 'services/api/types';

type EntityStateAndAdapter =
  | {
      id: string;
      type: CanvasLayerState['type'];
      state: CanvasLayerState;
      adapter: CanvasLayerAdapter;
    }
  | {
      id: string;
      type: CanvasInpaintMaskState['type'];
      state: CanvasInpaintMaskState;
      adapter: CanvasMaskAdapter;
    }
  // | {
  //     id: string;
  //     type: CanvasControlAdapterState['type'];
  //     state: CanvasControlAdapterState;
  //     adapter: CanvasControlAdapter;
  //   }
  | {
      id: string;
      type: CanvasRegionalGuidanceState['type'];
      state: CanvasRegionalGuidanceState;
      adapter: CanvasMaskAdapter;
    };

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

  getEntity(identifier: CanvasEntityIdentifier): EntityStateAndAdapter | null {
    const state = this.getState();

    let entityState: EntityStateAndAdapter['state'] | null = null;
    let entityAdapter: EntityStateAndAdapter['adapter'] | null = null;

    if (identifier.type === 'layer') {
      entityState = state.layers.entities.find((i) => i.id === identifier.id) ?? null;
      entityAdapter = this.manager.layers.get(identifier.id) ?? null;
    } else if (identifier.type === 'control_adapter') {
      entityState = state.controlAdapters.entities.find((i) => i.id === identifier.id) ?? null;
      entityAdapter = this.manager.controlAdapters.get(identifier.id) ?? null;
    } else if (identifier.type === 'regional_guidance') {
      entityState = state.regions.entities.find((i) => i.id === identifier.id) ?? null;
      entityAdapter = this.manager.regions.get(identifier.id) ?? null;
    } else if (identifier.type === 'inpaint_mask') {
      entityState = state.inpaintMask;
      entityAdapter = this.manager.inpaintMask;
    }

    if (entityState && entityAdapter) {
      return {
        id: entityState.id,
        type: entityState.type,
        state: entityState,
        adapter: entityAdapter,
      } as EntityStateAndAdapter; // TODO(psyche): make TS happy w/o this cast
    }

    return null;
  }

  getSelectedEntity = () => {
    const state = this.getState();
    if (state.selectedEntityIdentifier) {
      return this.getEntity(state.selectedEntityIdentifier);
    }
    return null;
  };

  getCurrentFill = () => {
    const state = this.getState();
    let currentFill: RgbaColor = state.tool.fill;
    const selectedEntity = this.getSelectedEntity();
    if (selectedEntity) {
      // These two entity types use a compositing rect for opacity. Their fill is always white.
      if (selectedEntity.state.type === 'regional_guidance' || selectedEntity.state.type === 'inpaint_mask') {
        currentFill = RGBA_RED;
        // currentFill = RGBA_WHITE;
      }
    }
    return currentFill;
  };

  getBrushPreviewFill = () => {
    const state = this.getState();
    let currentFill: RgbaColor = state.tool.fill;
    const selectedEntity = this.getSelectedEntity();
    if (selectedEntity) {
      // The brush should use the mask opacity for these entity types
      if (selectedEntity.state.type === 'regional_guidance' || selectedEntity.state.type === 'inpaint_mask') {
        currentFill = { ...selectedEntity.state.fill, a: this.getSettings().maskOpacity };
      }
    }
    return currentFill;
  };

  $transformingEntity: WritableAtom<CanvasEntityIdentifier | null> = atom();
  $toolState: WritableAtom<CanvasV2State['tool']> = atom();
  $currentFill: WritableAtom<RgbaColor> = atom();
  $selectedEntity: WritableAtom<EntityStateAndAdapter | null> = atom();
  $selectedEntityIdentifier: WritableAtom<CanvasEntityIdentifier | null> = atom();

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
