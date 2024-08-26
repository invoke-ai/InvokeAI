import { $alt, $ctrl, $meta, $shift } from '@invoke-ai/ui-library';
import type { AppStore } from 'app/store/store';
import type { CanvasLayerAdapter } from 'features/controlLayers/konva/CanvasLayerAdapter';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { CanvasMaskAdapter } from 'features/controlLayers/konva/CanvasMaskAdapter';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import {
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
} from 'features/controlLayers/store/canvasV2Slice';
import { selectAllRenderableEntities } from 'features/controlLayers/store/selectors';
import type {
  CanvasControlLayerState,
  CanvasEntityIdentifier,
  CanvasInpaintMaskState,
  CanvasRasterLayerState,
  CanvasRegionalGuidanceState,
  CanvasV2State,
  Coordinate,
  EntityBrushLineAddedPayload,
  EntityEraserLineAddedPayload,
  EntityIdentifierPayload,
  EntityMovedPayload,
  EntityRasterizedPayload,
  EntityRectAddedPayload,
  Rect,
  RgbaColor,
  RgbColor,
  StageAttrs,
  Tool,
} from 'features/controlLayers/store/types';
import { RGBA_BLACK } from 'features/controlLayers/store/types';
import type { WritableAtom } from 'nanostores';
import { atom } from 'nanostores';
import type { Logger } from 'roarr';
import { queueApi } from 'services/api/endpoints/queue';
import type { BatchConfig } from 'services/api/types';
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

export class CanvasStateApiModule extends CanvasModuleBase {
  readonly type = 'state_api';

  id: string;
  path: string[];
  manager: CanvasManager;
  log: Logger;
  subscriptions = new Set<() => void>();

  store: AppStore;

  constructor(store: AppStore, manager: CanvasManager) {
    super();
    this.id = getPrefixedId(this.type);
    this.manager = manager;
    this.path = this.manager.path.concat(this.id);
    this.log = this.manager.buildLogger(this.getLoggingContext);

    this.log.debug('Creating state api module');

    this.store = store;
  }

  // Reminder - use arrow functions to avoid binding issues
  getState = () => {
    return this.store.getState().canvasV2;
  };
  resetEntity = (arg: EntityIdentifierPayload) => {
    this.store.dispatch(entityReset(arg));
  };
  setEntityPosition = (arg: EntityMovedPayload) => {
    this.store.dispatch(entityMoved(arg));
  };
  addBrushLine = (arg: EntityBrushLineAddedPayload) => {
    this.store.dispatch(entityBrushLineAdded(arg));
  };
  addEraserLine = (arg: EntityEraserLineAddedPayload) => {
    this.store.dispatch(entityEraserLineAdded(arg));
  };
  addRect = (arg: EntityRectAddedPayload) => {
    this.store.dispatch(entityRectAdded(arg));
  };
  rasterizeEntity = (arg: EntityRasterizedPayload) => {
    this.store.dispatch(entityRasterized(arg));
  };
  setSelectedEntity = (arg: EntityIdentifierPayload) => {
    this.store.dispatch(entitySelected(arg));
  };
  setGenerationBbox = (bbox: Rect) => {
    this.store.dispatch(bboxChanged(bbox));
  };
  setBrushWidth = (width: number) => {
    this.store.dispatch(brushWidthChanged(width));
  };
  setEraserWidth = (width: number) => {
    this.store.dispatch(eraserWidthChanged(width));
  };
  setFill = (fill: RgbaColor) => {
    return this.store.dispatch(fillChanged(fill));
  };
  enqueueBatch = (batch: BatchConfig) => {
    this.store.dispatch(
      queueApi.endpoints.enqueueBatch.initiate(batch, {
        fixedCacheKey: 'enqueueBatch',
      })
    );
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

  getEntity(identifier: CanvasEntityIdentifier): EntityStateAndAdapter | null {
    const state = this.getState();

    let entityState: EntityStateAndAdapter['state'] | null = null;
    let entityAdapter: EntityStateAndAdapter['adapter'] | null = null;

    if (identifier.type === 'raster_layer') {
      entityState = state.rasterLayers.entities.find((i) => i.id === identifier.id) ?? null;
      entityAdapter = this.manager.adapters.rasterLayers.get(identifier.id) ?? null;
    } else if (identifier.type === 'control_layer') {
      entityState = state.controlLayers.entities.find((i) => i.id === identifier.id) ?? null;
      entityAdapter = this.manager.adapters.controlLayers.get(identifier.id) ?? null;
    } else if (identifier.type === 'regional_guidance') {
      entityState = state.regions.entities.find((i) => i.id === identifier.id) ?? null;
      entityAdapter = this.manager.adapters.regionMasks.get(identifier.id) ?? null;
    } else if (identifier.type === 'inpaint_mask') {
      entityState = state.inpaintMasks.entities.find((i) => i.id === identifier.id) ?? null;
      entityAdapter = this.manager.adapters.inpaintMasks.get(identifier.id) ?? null;
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

  getRenderedEntityCount = () => {
    const renderableEntities = selectAllRenderableEntities(this.getState());
    let count = 0;
    for (const entity of renderableEntities) {
      if (entity.isEnabled) {
        count++;
      }
    }
    return count;
  };

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
        currentFill = RGBA_BLACK;
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

  $transformingEntity = atom<CanvasEntityIdentifier | null>(null);
  $isProcessingTransform = atom<boolean>(false);

  $toolState: WritableAtom<CanvasV2State['tool']> = atom();
  $currentFill: WritableAtom<RgbaColor> = atom();
  $selectedEntity: WritableAtom<EntityStateAndAdapter | null> = atom();
  $selectedEntityIdentifier: WritableAtom<CanvasEntityIdentifier | null> = atom();
  $colorUnderCursor: WritableAtom<RgbColor> = atom(RGBA_BLACK);

  // Read-write state, ephemeral interaction state
  $tool = atom<Tool>('brush');
  $toolBuffer = atom<Tool | null>(null);
  $isDrawing = atom<boolean>(false);
  $isMouseDown = atom<boolean>(false);
  $lastAddedPoint = atom<Coordinate | null>(null);
  $lastMouseDownPos = atom<Coordinate | null>(null);
  $lastCursorPos = atom<Coordinate | null>(null);
  $lastCanvasProgressEvent = $lastCanvasProgressEvent;
  $spaceKey = atom<boolean>(false);
  $altKey = $alt;
  $ctrlKey = $ctrl;
  $metaKey = $meta;
  $shiftKey = $shift;
  $shouldShowStagedImage = atom(true);
  $stageAttrs = atom<StageAttrs>({
    x: 0,
    y: 0,
    width: 0,
    height: 0,
    scale: 0,
  });

  destroy = () => {
    this.log.debug('Destroying state api module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
  };

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
    };
  };

  getLoggingContext = () => {
    return { ...this.manager.getLoggingContext(), path: this.path.join('.') };
  };
}
