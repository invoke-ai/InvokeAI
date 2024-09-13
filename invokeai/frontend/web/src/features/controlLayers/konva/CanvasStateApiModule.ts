import { $alt, $ctrl, $meta, $shift } from '@invoke-ai/ui-library';
import type { Selector } from '@reduxjs/toolkit';
import { addAppListener } from 'app/store/middleware/listenerMiddleware';
import type { AppStore, RootState } from 'app/store/store';
import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import type { SubscriptionHandler } from 'features/controlLayers/konva/util';
import { createReduxSubscription, getPrefixedId } from 'features/controlLayers/konva/util';
import {
  selectCanvasSettingsSlice,
  settingsBrushWidthChanged,
  settingsColorChanged,
  settingsEraserWidthChanged,
} from 'features/controlLayers/store/canvasSettingsSlice';
import {
  $lastCanvasProgressEvent,
  bboxChangedFromCanvas,
  entityBrushLineAdded,
  entityEraserLineAdded,
  entityMoved,
  entityRasterized,
  entityRectAdded,
  entityReset,
} from 'features/controlLayers/store/canvasSlice';
import { selectCanvasStagingAreaSlice } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { selectAllRenderableEntities, selectBbox, selectCanvasSlice } from 'features/controlLayers/store/selectors';
import type {
  CanvasEntityType,
  CanvasState,
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
import { atom, computed } from 'nanostores';
import type { Logger } from 'roarr';
import { queueApi } from 'services/api/endpoints/queue';
import type { BatchConfig } from 'services/api/types';
import { assert } from 'tsafe';

import type { CanvasEntityAdapter } from './CanvasEntity/types';

export class CanvasStateApiModule extends CanvasModuleBase {
  readonly type = 'state_api';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasManager;
  readonly manager: CanvasManager;
  readonly log: Logger;

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
   * Runs a selector on the redux store.
   */
  runSelector = <T>(selector: Selector<RootState, T>) => {
    return selector(this.store.getState());
  };

  /**
   * Creates a subscription to the redux store.
   */
  createStoreSubscription = <T>(selector: Selector<RootState, T>, handler: SubscriptionHandler<T>) => {
    return createReduxSubscription(this.store, selector, handler);
  };

  /**
   * Adds a redux listener middleware listener.
   *
   * TODO(psyche): Unfortunately, this wrapper does not work correctly, due to a TS limitation.
   *
   * For a reason I do not understand, TS cannot resolve the parameter and return types of overloaded functions. It
   * only resolves one of the overload signatures.
   *
   * `addAppListener` has an overloaded type signature, so `Parameters<typeof addAppListener>[0]` resolves to only one
   * of the 5 possible arg types for the function. Unfortunately, you can't use this wrapper in the same way you could
   * if you called `addAppListener` directly.
   *
   * There are a number of proposed solutions but none worked for me. I think there may be limitations with the use of
   * generics? See:
   * - https://github.com/microsoft/TypeScript/issues/32164
   * - https://github.com/microsoft/TypeScript/issues/29732
   */
  addStoreListener = (arg: Parameters<typeof addAppListener>[0]) => {
    return this.store.dispatch(addAppListener(arg));
  };

  /**
   * Gets the canvas slice.
   *
   * The state is stored in redux.
   */
  getCanvasState = (): CanvasState => {
    return this.runSelector(selectCanvasSlice);
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
    this.store.dispatch(bboxChangedFromCanvas(rect));
  };

  /**
   * Sets the brush width, pushing state to redux.
   */
  setBrushWidth = (width: number) => {
    this.store.dispatch(settingsBrushWidthChanged(width));
  };

  /**
   * Sets the eraser width, pushing state to redux.
   */
  setEraserWidth = (width: number) => {
    this.store.dispatch(settingsEraserWidthChanged(width));
  };

  /**
   * Sets the drawing color, pushing state to redux.
   */
  setColor = (color: RgbaColor) => {
    return this.store.dispatch(settingsColorChanged(color));
  };

  /**
   * Enqueues a batch, pushing state to redux.
   */
  enqueueBatch = (batch: BatchConfig) => {
    return this.store.dispatch(
      queueApi.endpoints.enqueueBatch.initiate(batch, {
        fixedCacheKey: 'enqueueBatch',
      })
    );
  };

  /**
   * Gets the generation bbox state from redux.
   */
  getBbox = () => {
    return this.runSelector(selectBbox);
  };

  /**
   * Gets the canvas settings from redux.
   */
  getSettings = () => {
    return this.runSelector(selectCanvasSettingsSlice);
  };

  getGridSize = (): number => {
    const snapToGrid = this.getSettings().snapToGrid;
    if (!snapToGrid) {
      return 1;
    }
    const useFine = this.$ctrlKey.get() || this.$metaKey.get();
    if (useFine) {
      return 8;
    }
    return 64;
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
   * Gets the canvas staging area state from redux.
   */
  getStagingArea = () => {
    return this.runSelector(selectCanvasStagingAreaSlice);
  };

  /**
   * Checks if an entity is selected.
   */
  getIsSelected = (id: string): boolean => {
    return this.getCanvasState().selectedEntityIdentifier?.id === id;
  };

  /**
   * Checks if an entity type is hidden. Individual entities are not hidden; the entire entity type is hidden.
   */
  getIsTypeHidden = (type: CanvasEntityType): boolean => {
    switch (type) {
      case 'raster_layer':
        return this.getRasterLayersState().isHidden;
      case 'control_layer':
        return this.getControlLayersState().isHidden;
      case 'inpaint_mask':
        return this.getInpaintMasksState().isHidden;
      case 'regional_guidance':
        return this.getRegionsState().isHidden;
      default:
        assert(false, 'Unhandled entity type');
    }
  };

  /**
   * Gets the number of entities that are currently rendered on the canvas.
   */
  getRenderedEntityCount = (): number => {
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
   * Gets the currently selected entity's adapter
   */
  getSelectedEntityAdapter = (): CanvasEntityAdapter | null => {
    const state = this.getCanvasState();
    if (state.selectedEntityIdentifier) {
      return this.manager.getAdapter(state.selectedEntityIdentifier);
    }
    return null;
  };

  /**
   * Gets the current drawing color.
   *
   * The color is determined by the tool state, except when the selected entity is a regional guidance or inpaint mask.
   * In that case, the color is always black.
   *
   * Regional guidance and inpaint mask entities use a compositing rect to draw with their selected color and texture,
   * so the color for lines and rects doesn't matter - it is never seen. The only requirement is that it is opaque. For
   * consistency with conventional black and white mask images, we use black as the color for these entities.
   */
  getCurrentColor = (): RgbaColor => {
    let color: RgbaColor = this.getSettings().color;
    const selectedEntity = this.getSelectedEntityAdapter();
    if (selectedEntity) {
      // These two entity types use a compositing rect for opacity. Their fill is always a solid color.
      if (selectedEntity.state.type === 'regional_guidance' || selectedEntity.state.type === 'inpaint_mask') {
        color = RGBA_BLACK;
      }
    }
    return color;
  };

  /**
   * Gets the brush preview color. The brush preview color is determined by the tool state and the selected entity.
   *
   * The color is the tool state's color, except when the selected entity is a regional guidance or inpaint mask.
   *
   * These entities have their own color and texture, so the brush preview should use those instead of the tool state's
   * color.
   */
  getBrushPreviewColor = (): RgbaColor => {
    const selectedEntity = this.getSelectedEntityAdapter();
    if (selectedEntity?.state.type === 'regional_guidance' || selectedEntity?.state.type === 'inpaint_mask') {
      // TODO(psyche): If we move the brush preview's Konva nodes to the selected entity renderer, we can draw them
      // under the entity's compositing rect, so they would use selected entity's selected color and texture. As a
      // temporary workaround to improve the UX when using a brush on a regional guidance or inpaint mask, we use the
      // selected entity's fill color with 50% opacity.
      return { ...selectedEntity.state.fill.color, a: 0.5 };
    } else {
      return this.getSettings().color;
    }
  };

  /**
   * The entity adapter being filtered, if any.
   */
  $filteringAdapter = atom<CanvasEntityAdapterRasterLayer | CanvasEntityAdapterControlLayer | null>(null);

  /**
   * Whether an entity is currently being filtered. Derived from `$filteringAdapter`.
   */
  $isFiltering = computed(this.$filteringAdapter, (filteringAdapter) => Boolean(filteringAdapter));

  /**
   * The entity adapter being transformed, if any.
   */
  $transformingAdapter = atom<CanvasEntityAdapter | null>(null);

  /**
   * Whether an entity is currently being transformed. Derived from `$transformingAdapter`.
   */
  $isTranforming = computed(this.$transformingAdapter, (transformingAdapter) => Boolean(transformingAdapter));

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
