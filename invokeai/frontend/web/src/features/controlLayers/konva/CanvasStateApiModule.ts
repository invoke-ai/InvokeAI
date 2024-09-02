import { $alt, $ctrl, $meta, $shift } from '@invoke-ai/ui-library';
import type { AppStore } from 'app/store/store';
import type { CanvasEntityLayerAdapter } from 'features/controlLayers/konva/CanvasEntityLayerAdapter';
import type { CanvasEntityMaskAdapter } from 'features/controlLayers/konva/CanvasEntityMaskAdapter';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import {
  bboxChanged,
  entityBrushLineAdded,
  entityEraserLineAdded,
  entityMoved,
  entityRasterized,
  entityRectAdded,
  entityReset,
} from 'features/controlLayers/store/canvasSlice';
import { selectAllRenderableEntities, selectCanvasSlice } from 'features/controlLayers/store/selectors';
import {
  brushWidthChanged,
  eraserWidthChanged,
  fillChanged,
  type ToolState,
} from 'features/controlLayers/store/toolSlice';
import type {
  CanvasControlLayerState,
  CanvasEntityIdentifier,
  CanvasInpaintMaskState,
  CanvasRasterLayerState,
  CanvasRegionalGuidanceState,
  EntityBrushLineAddedPayload,
  EntityEraserLineAddedPayload,
  EntityIdentifierPayload,
  EntityMovedPayload,
  EntityRasterizedPayload,
  EntityRectAddedPayload,
  Rect,
  RgbaColor,
} from 'features/controlLayers/store/types';
import { RGBA_BLACK } from 'features/controlLayers/store/types';
import type { WritableAtom } from 'nanostores';
import { atom, computed } from 'nanostores';
import type { Logger } from 'roarr';
import { queueApi } from 'services/api/endpoints/queue';
import type { BatchConfig } from 'services/api/types';
import { $lastCanvasProgressEvent } from 'services/events/setEventListeners';

type EntityStateAndAdapter =
  | {
      id: string;
      type: CanvasRasterLayerState['type'];
      state: CanvasRasterLayerState;
      adapter: CanvasEntityLayerAdapter;
    }
  | {
      id: string;
      type: CanvasControlLayerState['type'];
      state: CanvasControlLayerState;
      adapter: CanvasEntityLayerAdapter;
    }
  | {
      id: string;
      type: CanvasInpaintMaskState['type'];
      state: CanvasInpaintMaskState;
      adapter: CanvasEntityMaskAdapter;
    }
  | {
      id: string;
      type: CanvasRegionalGuidanceState['type'];
      state: CanvasRegionalGuidanceState;
      adapter: CanvasEntityMaskAdapter;
    };

export class CanvasStateApiModule extends CanvasModuleBase {
  readonly type = 'state_api';

  id: string;
  path: string[];
  parent: CanvasManager;
  manager: CanvasManager;
  log: Logger;

  /**
   * The redux store.
   */
  store: AppStore;

  constructor(store: AppStore, manager: CanvasManager) {
    super();
    this.id = getPrefixedId(this.type);
    this.parent = manager;
    this.manager = manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug('Creating state api module');

    this.store = store;
  }

  /**
   * Gets the canvas slice.
   *
   * The state is stored in redux.
   */
  getCanvasState = () => {
    return selectCanvasSlice(this.store.getState());
  };

  /**
   * Resets an entity, pushing state to redux.
   */
  resetEntity = (arg: EntityIdentifierPayload) => {
    this.store.dispatch(entityReset(arg));
  };

  /**
   * Updates an entity's position, pushing state to redux.
   */
  setEntityPosition = (arg: EntityMovedPayload) => {
    this.store.dispatch(entityMoved(arg));
  };

  /**
   * Adds a brush line to an entity, pushing state to redux.
   */
  addBrushLine = (arg: EntityBrushLineAddedPayload) => {
    this.store.dispatch(entityBrushLineAdded(arg));
  };

  /**
   * Adds an eraser line to an entity, pushing state to redux.
   */
  addEraserLine = (arg: EntityEraserLineAddedPayload) => {
    this.store.dispatch(entityEraserLineAdded(arg));
  };

  /**
   * Adds a rectangle to an entity, pushing state to redux.
   */
  addRect = (arg: EntityRectAddedPayload) => {
    this.store.dispatch(entityRectAdded(arg));
  };

  /**
   * Rasterizes an entity, pushing state to redux.
   */
  rasterizeEntity = (arg: EntityRasterizedPayload) => {
    this.store.dispatch(entityRasterized(arg));
  };

  /**
   * Sets the generation bbox rect, pushing state to redux.
   */
  setGenerationBbox = (rect: Rect) => {
    this.store.dispatch(bboxChanged(rect));
  };

  /**
   * Sets the brush width, pushing state to redux.
   */
  setBrushWidth = (width: number) => {
    this.store.dispatch(brushWidthChanged(width));
  };

  /**
   * Sets the eraser width, pushing state to redux.
   */
  setEraserWidth = (width: number) => {
    this.store.dispatch(eraserWidthChanged(width));
  };

  /**
   * Sets the fill color, pushing state to redux.
   */
  setFill = (fill: RgbaColor) => {
    return this.store.dispatch(fillChanged(fill));
  };

  /**
   * Enqueues a batch, pushing state to redux.
   */
  enqueueBatch = (batch: BatchConfig) => {
    this.store.dispatch(
      queueApi.endpoints.enqueueBatch.initiate(batch, {
        fixedCacheKey: 'enqueueBatch',
      })
    );
  };

  /**
   * Gets the generation bbox state from redux.
   */
  getBbox = () => {
    return this.getCanvasState().bbox;
  };

  /**
   * Gets the tool state from redux.
   */
  getToolState = () => {
    return this.store.getState().tool;
  };

  /**
   * Gets the canvas settings from redux.
   */
  getSettings = () => {
    return this.store.getState().canvasSettings;
  };

  /**
   * Gets the regions state from redux.
   */
  getRegionsState = () => {
    return this.getCanvasState().regions;
  };

  /**
   * Gets the raster layers state from redux.
   */
  getRasterLayersState = () => {
    return this.getCanvasState().rasterLayers;
  };

  /**
   * Gets the control layers state from redux.
   */
  getControlLayersState = () => {
    return this.getCanvasState().controlLayers;
  };

  /**
   * Gets the inpaint masks state from redux.
   */
  getInpaintMasksState = () => {
    return this.getCanvasState().inpaintMasks;
  };

  /**
   * Gets the canvas session state from redux.
   */
  getSession = () => {
    return this.store.getState().canvasSession;
  };

  /**
   * Checks if an entity is selected.
   */
  getIsSelected = (id: string): boolean => {
    return this.getCanvasState().selectedEntityIdentifier?.id === id;
  };

  /**
   * Gets an entity by its identifier. The entity's state is retrieved from the redux store, and its adapter is
   * retrieved from the canvas manager.
   *
   * Both state and adapter must exist for the entity to be returned.
   */
  getEntity(identifier: CanvasEntityIdentifier): EntityStateAndAdapter | null {
    const state = this.getCanvasState();

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

  /**
   * Gets the number of entities that are currently rendered on the canvas.
   */
  getRenderedEntityCount = () => {
    const renderableEntities = selectAllRenderableEntities(this.getCanvasState());
    let count = 0;
    for (const entity of renderableEntities) {
      if (entity.isEnabled) {
        count++;
      }
    }
    return count;
  };

  /**
   * Gets the currently selected entity, if any. The entity's state is retrieved from the redux store, and its adapter
   * is retrieved from the canvas manager.
   */
  getSelectedEntity = () => {
    const state = this.getCanvasState();
    if (state.selectedEntityIdentifier) {
      return this.getEntity(state.selectedEntityIdentifier);
    }
    return null;
  };

  /**
   * Gets the current fill color. The fill color is determined by the tool state and the selected entity.
   *
   * The fill color is determined by the tool state, except when the selected entity is a regional guidance or inpaint
   * mask. In that case, the fill color is always black.
   *
   * Regional guidance and inpaint mask entities use a compositing rect to draw with their selected color and texture,
   * so the fill color for lines and rects doesn't matter - it is never seen. The only requirement is that it is opaque.
   * For consistency with conventional black and white mask images, we use black as the fill color for these entities.
   */
  getCurrentFill = () => {
    let currentFill: RgbaColor = this.getToolState().fill;
    const selectedEntity = this.getSelectedEntity();
    if (selectedEntity) {
      // These two entity types use a compositing rect for opacity. Their fill is always a solid color.
      if (selectedEntity.state.type === 'regional_guidance' || selectedEntity.state.type === 'inpaint_mask') {
        currentFill = RGBA_BLACK;
      }
    }
    return currentFill;
  };

  /**
   * Gets the brush preview fill color. The brush preview fill color is determined by the tool state and the selected
   * entity.
   *
   * The color is the tool state's fill color, except when the selected entity is a regional guidance or inpaint mask.
   *
   * These entities have their own fill color and texture, so the brush preview should use those instead of the tool
   * state's fill color.
   */
  getBrushPreviewFill = (): RgbaColor => {
    const selectedEntity = this.getSelectedEntity();
    if (selectedEntity?.state.type === 'regional_guidance' || selectedEntity?.state.type === 'inpaint_mask') {
      // TODO(psyche): If we move the brush preview's Konva nodes to the selected entity renderer, we can draw them
      // under the entity's compositing rect, so they would use selected entity's selected color and texture. As a
      // temporary workaround to improve the UX when using a brush on a regional guidance or inpaint mask, we use the
      // selected entity's fill color with 50% opacity.
      return { ...selectedEntity.state.fill.color, a: 0.5 };
    } else {
      return this.getToolState().fill;
    }
  };

  /**
   * The entity adapter being transformed, if any.
   */
  $transformingAdapter = atom<CanvasEntityLayerAdapter | CanvasEntityMaskAdapter | null>(null);

  /**
   * Whether an entity is currently being transformed. Derived from `$transformingAdapter`.
   */
  $isTranforming = computed(this.$transformingAdapter, (transformingAdapter) => Boolean(transformingAdapter));

  /**
   * A nanostores atom, kept in sync with the redux store's tool state.
   */
  $toolState: WritableAtom<ToolState> = atom();

  /**
   * The current fill color, derived from the tool state and the selected entity.
   */
  $currentFill: WritableAtom<RgbaColor> = atom();

  /**
   * The currently selected entity, if any. Includes the entity latest state and its adapter.
   */
  $selectedEntity: WritableAtom<EntityStateAndAdapter | null> = atom();

  /**
   * The currently selected entity's identifier, if an entity is selected.
   */
  $selectedEntityIdentifier: WritableAtom<CanvasEntityIdentifier | null> = atom();

  /**
   * The last canvas progress event. This is set in a global event listener. The staging area may set it to null when it
   * consumes the event.
   */
  $lastCanvasProgressEvent = $lastCanvasProgressEvent;

  /**
   * Whether the space key is currently pressed.
   */
  $spaceKey = atom<boolean>(false);

  /**
   * Whether the alt key is currently pressed.
   */
  $altKey = $alt;

  /**
   * Whether the ctrl key is currently pressed.
   */
  $ctrlKey = $ctrl;

  /**
   * Whether the meta key is currently pressed.
   */
  $metaKey = $meta;

  /**
   * Whether the shift key is currently pressed.
   */
  $shiftKey = $shift;
}
