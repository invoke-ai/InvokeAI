import { $alt, $ctrl, $meta, $shift } from '@invoke-ai/ui-library';
import type { AppStore } from 'app/store/store';
import type { CanvasLayerAdapter } from 'features/controlLayers/konva/CanvasLayerAdapter';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { CanvasMaskAdapter } from 'features/controlLayers/konva/CanvasMaskAdapter';
import {
  $filterConfig,
  $filteringEntity,
  $isDrawing,
  $isMouseDown,
  $isProcessingFilter,
  $lastAddedPoint,
  $lastCursorPos,
  $lastMouseDownPos,
  $shouldShowStagedImage,
  $spaceKey,
  $stageAttrs,
  $transformingEntity,
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
  fillChanged,
  inpaintMaskCompositeRasterized,
  rasterLayerCompositeRasterized,
  toolBufferChanged,
  toolChanged,
} from 'features/controlLayers/store/canvasV2Slice';
import type {
  CanvasControlLayerState,
  CanvasEntityIdentifier,
  CanvasInpaintMaskState,
  CanvasRasterLayerState,
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
import { $lastCanvasProgressEvent } from 'services/events/setEventListeners';

type EntityStateAndAdapter =
  | {
      id: string;
      type: CanvasRasterLayerState['type'];
      state: CanvasRasterLayerState;
      adapter: CanvasLayerAdapter;
    }
  | {
      id: string;
      type: CanvasControlLayerState['type'];
      state: CanvasControlLayerState;
      adapter: CanvasLayerAdapter;
    }
  | {
      id: string;
      type: CanvasInpaintMaskState['type'];
      state: CanvasInpaintMaskState;
      adapter: CanvasMaskAdapter;
    }
  | {
      id: string;
      type: CanvasRegionalGuidanceState['type'];
      state: CanvasRegionalGuidanceState;
      adapter: CanvasMaskAdapter;
    };

export class CanvasStateApi {
  _store: AppStore;
  manager: CanvasManager;

  constructor(store: AppStore, manager: CanvasManager) {
    this._store = store;
    this.manager = manager;
  }

  // Reminder - use arrow functions to avoid binding issues
  getState = () => {
    return this._store.getState().canvasV2;
  };
  resetEntity = (arg: EntityIdentifierPayload) => {
    this._store.dispatch(entityReset(arg));
  };
  setEntityPosition = (arg: EntityMovedPayload) => {
    this._store.dispatch(entityMoved(arg));
  };
  addBrushLine = (arg: EntityBrushLineAddedPayload) => {
    this._store.dispatch(entityBrushLineAdded(arg));
  };
  addEraserLine = (arg: EntityEraserLineAddedPayload) => {
    this._store.dispatch(entityEraserLineAdded(arg));
  };
  addRect = (arg: EntityRectAddedPayload) => {
    this._store.dispatch(entityRectAdded(arg));
  };
  rasterizeEntity = (arg: EntityRasterizedPayload) => {
    this._store.dispatch(entityRasterized(arg));
  };
  compositeRasterLayerRasterized = (arg: { imageName: string; rect: Rect }) => {
    this._store.dispatch(rasterLayerCompositeRasterized(arg));
  };
  compositeInpaintMaskRasterized = (arg: { imageName: string; rect: Rect }) => {
    this._store.dispatch(inpaintMaskCompositeRasterized(arg));
  };
  setSelectedEntity = (arg: EntityIdentifierPayload) => {
    this._store.dispatch(entitySelected(arg));
  };
  setGenerationBbox = (bbox: Rect) => {
    this._store.dispatch(bboxChanged(bbox));
  };
  setBrushWidth = (width: number) => {
    this._store.dispatch(brushWidthChanged(width));
  };
  setEraserWidth = (width: number) => {
    this._store.dispatch(eraserWidthChanged(width));
  };
  setTool = (tool: Tool) => {
    this._store.dispatch(toolChanged(tool));
  };
  setToolBuffer = (toolBuffer: Tool | null) => {
    this._store.dispatch(toolBufferChanged(toolBuffer));
  };
  setFill = (fill: RgbaColor) => {
    return this._store.dispatch(fillChanged(fill));
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
  getRasterLayersState = () => {
    return this.getState().rasterLayers;
  };
  getControlLayersState = () => {
    return this.getState().controlLayers;
  };
  getInpaintMasksState = () => {
    return this.getState().inpaintMasks;
  };
  getSession = () => {
    return this.getState().session;
  };
  getIsSelected = (id: string) => {
    return this.getState().selectedEntityIdentifier?.id === id;
  };
  getFilterState = () => {
    return this._store.getState().canvasV2.filter;
  };

  getEntity(identifier: CanvasEntityIdentifier): EntityStateAndAdapter | null {
    const state = this.getState();

    let entityState: EntityStateAndAdapter['state'] | null = null;
    let entityAdapter: EntityStateAndAdapter['adapter'] | null = null;

    if (identifier.type === 'raster_layer') {
      entityState = state.rasterLayers.entities.find((i) => i.id === identifier.id) ?? null;
      entityAdapter = this.manager.rasterLayerAdapters.get(identifier.id) ?? null;
    } else if (identifier.type === 'control_layer') {
      entityState = state.controlLayers.entities.find((i) => i.id === identifier.id) ?? null;
      entityAdapter = this.manager.controlLayerAdapters.get(identifier.id) ?? null;
    } else if (identifier.type === 'regional_guidance') {
      entityState = state.regions.entities.find((i) => i.id === identifier.id) ?? null;
      entityAdapter = this.manager.regionalGuidanceAdapters.get(identifier.id) ?? null;
    } else if (identifier.type === 'inpaint_mask') {
      entityState = state.inpaintMasks.entities.find((i) => i.id === identifier.id) ?? null;
      entityAdapter = this.manager.inpaintMaskAdapters.get(identifier.id) ?? null;
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
      // These two entity types use a compositing rect for opacity. Their fill is always a solid color.
      if (selectedEntity.state.type === 'regional_guidance' || selectedEntity.state.type === 'inpaint_mask') {
        currentFill = RGBA_RED;
      }
    }
    return currentFill;
  };

  getBrushPreviewFill = (): RgbaColor => {
    const selectedEntity = this.getSelectedEntity();
    if (selectedEntity?.state.type === 'regional_guidance' || selectedEntity?.state.type === 'inpaint_mask') {
      // The brush should use the mask opacity for these enktity types
      return { ...selectedEntity.state.fill.color, a: 1 };
    } else {
      const state = this.getState();
      return state.tool.fill;
    }
  };

  $transformingEntity = $transformingEntity;
  $filteringEntity = $filteringEntity;
  $filterConfig = $filterConfig;
  $isProcessingFilter = $isProcessingFilter;

  $toolState: WritableAtom<CanvasV2State['tool']> = atom();
  $currentFill: WritableAtom<RgbaColor> = atom();
  $selectedEntity: WritableAtom<EntityStateAndAdapter | null> = atom();
  $selectedEntityIdentifier: WritableAtom<CanvasEntityIdentifier | null> = atom();
  $colorUnderCursor: WritableAtom<RgbaColor | null> = atom();

  // Read-write state, ephemeral interaction state
  $isDrawing = $isDrawing;
  $isMouseDown = $isMouseDown;
  $lastAddedPoint = $lastAddedPoint;
  $lastMouseDownPos = $lastMouseDownPos;
  $lastCursorPos = $lastCursorPos;
  $lastCanvasProgressEvent = $lastCanvasProgressEvent;
  $spaceKey = $spaceKey;
  $altKey = $alt;
  $ctrlKey = $ctrl;
  $metaKey = $meta;
  $shiftKey = $shift;
  $shouldShowStagedImage = $shouldShowStagedImage;
  $stageAttrs = $stageAttrs;
}
