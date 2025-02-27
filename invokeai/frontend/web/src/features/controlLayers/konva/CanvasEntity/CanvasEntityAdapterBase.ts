import type { Selector } from '@reduxjs/toolkit';
import { createSelector } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import { withResultAsync } from 'common/util/result';
import type { CanvasEntityBufferObjectRenderer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityBufferObjectRenderer';
import type { CanvasEntityFilterer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityFilterer';
import type { CanvasEntityObjectRenderer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityObjectRenderer';
import type { CanvasEntityTransformer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityTransformer';
import type { CanvasEntityAdapter } from 'features/controlLayers/konva/CanvasEntity/types';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import type { CanvasSegmentAnythingModule } from 'features/controlLayers/konva/CanvasSegmentAnythingModule';
import { getKonvaNodeDebugAttrs, getRectIntersection } from 'features/controlLayers/konva/util';
import {
  selectIsolatedLayerPreview,
  selectIsolatedStagingPreview,
} from 'features/controlLayers/store/canvasSettingsSlice';
import { selectIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import {
  buildSelectIsSelected,
  getSelectIsTypeHidden,
  selectBboxRect,
  selectCanvasSlice,
  selectEntity,
} from 'features/controlLayers/store/selectors';
import {
  type CanvasEntityIdentifier,
  type CanvasRenderableEntityState,
  isRasterLayerEntityIdentifier,
  type Rect,
} from 'features/controlLayers/store/types';
import { toast } from 'features/toast/toast';
import Konva from 'konva';
import { atom } from 'nanostores';
import rafThrottle from 'raf-throttle';
import type { Logger } from 'roarr';
import type { ImageDTO } from 'services/api/types';
import stableHash from 'stable-hash';
import { assert } from 'tsafe';
import type { Jsonifiable, JsonObject } from 'type-fest';

// Ideally, we'd type `adapter` as `CanvasEntityAdapterBase`, but the generics make this tricky. `CanvasEntityAdapter`
// is a union of all entity adapters and is functionally identical to `CanvasEntityAdapterBase`. We'll need to do a
// type assertion below in the `onInit` method, which calls these callbacks.
type InitCallback = (adapter: CanvasEntityAdapter) => Promise<boolean>;

export abstract class CanvasEntityAdapterBase<
  T extends CanvasRenderableEntityState,
  U extends string,
> extends CanvasModuleBase {
  readonly type: U;
  readonly id: string;
  readonly path: string[];
  readonly manager: CanvasManager;
  readonly parent: CanvasManager;
  readonly log: Logger;

  readonly entityIdentifier: CanvasEntityIdentifier<T['type']>;

  /**
   * The transformer for this entity adapter. All entities must have a transformer.
   */
  abstract transformer: CanvasEntityTransformer;

  /**
   * The renderer for this entity adapter. All entities must have a renderer.
   */
  abstract renderer: CanvasEntityObjectRenderer;

  /**
   * The buffer renderer for this entity adapter. All entities must have a buffer renderer.
   */
  abstract bufferRenderer: CanvasEntityBufferObjectRenderer;

  /**
   * The filterer for this entity adapter. Entities that support filtering should implement this property.
   */
  // TODO(psyche): This is in the ABC and not in the concrete classes to allow all adapters to share the `destroy`
  // method. If it wasn't in this ABC, we'd get a TS error in `destroy`. Maybe there's a better way to handle this
  // without requiring all adapters to implement this property and their own `destroy`?
  abstract filterer?: CanvasEntityFilterer;

  /**
   * The segment anything module for this entity adapter. Entities that support segment anything should implement
   * this property.
   */
  // TODO(psyche): This is in the ABC and not in the concrete classes to allow all adapters to share the `destroy`
  // method. If it wasn't in this ABC, we'd get a TS error in `destroy`. Maybe there's a better way to handle this
  // without requiring all adapters to implement this property and their own `destroy`?
  abstract segmentAnything?: CanvasSegmentAnythingModule;

  /**
   * Synchronizes the entity state with the canvas. This includes rendering the entity's objects, handling visibility,
   * positioning, opacity, locked state, and any other properties.
   *
   * Implementations should be minimal and should only update the canvas if the state has changed.
   *
   * If `state` is undefined, the entity was just deleted and the adapter should destroy itself.
   *
   * If `prevState` is undefined, this is the first time the entity is being synced.
   */
  abstract sync: (state: T | undefined, prevState: T | undefined) => Promise<void>;

  /**
   * Gets the canvas element for the entity. If `rect` is provided, the canvas will be clipped to that rectangle.
   */
  abstract getCanvas: (rect?: Rect) => HTMLCanvasElement;

  /**
   * Gets a hashable representation of the entity's _renderable_ state. This should exclude any properties that are not
   * relevant to rendering the entity.
   *
   * This is used for caching.
   */
  abstract getHashableState: () => JsonObject;

  /**
   * Callbacks that are executed when the module is initialized.
   */
  private static initCallbacks = new Set<InitCallback>();

  /**
   * Register a callback to be run when an entity adapter is initialized.
   *
   * The callback is called for every adapter that is initialized with the adapter as its only argument. Use an early
   * return to skip entities that are not of interest, returning `false` to keep the callback registered. Return `true`
   * to unregister the callback after it is called.
   *
   * @param callback The callback to register.
   *
   * @example
   * ```ts
   * // A callback that is executed once for a specific entity:
   * const myId = 'my_id';
   * canvasManager.entityRenderer.registerOnInitCallback(async (adapter) => {
   *   if (adapter.id !== myId) {
   *     // These are not the droids you are looking for, move along
   *     return false;
   *   }
   *
   *   doSomething();
   *
   *   // Remove the callback
   *   return true;
   * });
   * ```
   *
   * @example
   * ```ts
   * // A callback that is executed once for the next entity that is initialized:
   * canvasManager.entityRenderer.registerOnInitCallback(async (adapter) => {
   *   doSomething();
   *
   *   // Remove the callback
   *   return true;
   * });
   * ```
   *
   * @example
   * ```ts
   * // A callback that is executed for every entity and is never removed:
   * canvasManager.entityRenderer.registerOnInitCallback(async (adapter) => {
   *   // Do something with the adapter
   *   return false;
   * });
   */
  static registerInitCallback = (callback: InitCallback) => {
    const wrapped = async (adapter: CanvasEntityAdapter) => {
      const result = await callback(adapter);
      if (result) {
        this.initCallbacks.delete(wrapped);
      }
      return result;
    };
    this.initCallbacks.add(wrapped);
  };

  /**
   * Runs all init callbacks with the given entity adapter.
   * @param adapter The adapter of the entity that was initialized.
   */
  private static runInitCallbacks = (adapter: CanvasEntityAdapter) => {
    for (const callback of this.initCallbacks) {
      callback(adapter);
    }
  };

  /**
   * A selector that selects whether the entity type is hidden.
   */
  selectIsTypeHidden: Selector<RootState, boolean>;

  /**
   * A selector that selects whether the entity is selected.
   */
  selectIsSelected: Selector<RootState, boolean>;

  /**
   * The Konva nodes that make up the entity adapter:
   * - A Konva.Layer to hold the everything
   *
   * Note that the transformer and object renderer have their own Konva nodes, but they are not stored here.
   */
  konva: {
    layer: Konva.Layer;
  };

  /**
   * The entity's state.
   */
  state: T;

  /**
   * A set of subscriptions to stores.
   */
  subscriptions = new Set<() => void>();

  /**
   * Whether this entity is locked. This is synced with the entity's state.
   */
  $isLocked = atom(false);
  /**
   * Whether this entity is disabled. This is synced with the entity's state.
   */
  $isDisabled = atom(false);
  /**
   * Whether this entity is hidden. This is synced with the entity's group type visibility.
   */
  $isEntityTypeHidden = atom(false);
  /**
   * Whether this entity is empty. This is computed based on the entity's objects.
   */
  $isEmpty = atom(true);
  /**
   * A cache of the entity's canvas element. This is generated from a clone of the entity's Konva layer.
   */
  $canvasCache = atom<HTMLCanvasElement | null>(null);
  /**
   * Whether this entity is onscreen. This is computed based on the entity's bounding box and the stage's viewport rect.
   */
  $isOnScreen = atom(true);
  /**
   * Whether this entity's rect intersects the bbox rect.
   */
  $intersectsBbox = atom(false);

  constructor(entityIdentifier: CanvasEntityIdentifier<T['type']>, manager: CanvasManager, adapterType: U) {
    super();
    this.type = adapterType;
    this.id = entityIdentifier.id;
    this.entityIdentifier = entityIdentifier;
    this.manager = manager;
    this.parent = manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);

    this.log.debug('Creating module');

    this.konva = {
      layer: new Konva.Layer({
        name: `${this.type}:layer`,
        listening: false,
        imageSmoothingEnabled: false,
      }),
    };

    this.manager.stage.addLayer(this.konva.layer);

    // We must have the entity state on creation.
    const state = this.manager.stateApi.runSelector(this.selectState);
    assert(state !== undefined, 'Missing entity state on creation');
    this.state = state;

    this.selectIsTypeHidden = getSelectIsTypeHidden(this.entityIdentifier.type);
    this.selectIsSelected = buildSelectIsSelected(this.entityIdentifier);

    /**
     * There are a number of reason we may need to show or hide a layer:
     * - The entity is enabled/disabled
     * - The entity type is hidden/shown
     * - `isolatedStagingPreview` is enabled and we start or stop staging
     * - `isolatedLayerPreview` is enabled and we start or stop filtering, transforming, select-object-ing
     * - The entity is selected or deselected (only selected and onscreen entities are rendered as a perf optimization)
     */
    this.subscriptions.add(this.manager.stateApi.createStoreSubscription(this.selectIsTypeHidden, this.syncVisibility));
    this.subscriptions.add(
      this.manager.stateApi.createStoreSubscription(selectIsolatedLayerPreview, this.syncVisibility)
    );
    this.subscriptions.add(
      this.manager.stateApi.createStoreSubscription(selectIsolatedStagingPreview, this.syncVisibility)
    );
    this.subscriptions.add(this.manager.stateApi.createStoreSubscription(selectIsStaging, this.syncVisibility));
    this.subscriptions.add(this.manager.stateApi.$filteringAdapter.listen(this.syncVisibility));
    this.subscriptions.add(this.manager.stateApi.$transformingAdapter.listen(this.syncVisibility));
    this.subscriptions.add(this.manager.stateApi.$segmentingAdapter.listen(this.syncVisibility));
    this.subscriptions.add(this.manager.stateApi.createStoreSubscription(this.selectIsSelected, this.syncVisibility));

    /**
     * The tool preview may need to be updated when the entity is locked or disabled. For example, when we disable the
     * entity, we should hide the tool preview & change the cursor.
     */
    this.subscriptions.add(this.$isDisabled.subscribe(this.manager.tool.render));
    this.subscriptions.add(this.$isLocked.subscribe(this.manager.tool.render));
    this.subscriptions.add(this.$isEntityTypeHidden.subscribe(this.manager.tool.render));

    /**
     * When the stage is transformed in any way (panning, zooming, resizing) or the entity is moved, we need to update
     * the entity's onscreen status. We also need to subscribe to changes to the entity's pixel rect, but this is
     * handled in the initialize method.
     */
    this.subscriptions.add(this.manager.stage.$stageAttrs.listen(this.syncIsOnscreen));
    this.subscriptions.add(this.manager.stateApi.createStoreSubscription(this.selectPosition, this.syncIsOnscreen));

    /**
     * When the bbox rect changes or the entity is moved, we need to update the intersectsBbox status. We also need to
     * subscribe to changes to the entity's pixel rect, but this is handled in the initialize method.
     */
    this.subscriptions.add(this.manager.stateApi.createStoreSubscription(selectBboxRect, this.syncIntersectsBbox));
    this.subscriptions.add(this.manager.stateApi.createStoreSubscription(this.selectPosition, this.syncIntersectsBbox));
  }

  /**
   * A redux selector that selects the entity's state from the canvas slice.
   */
  selectState = createSelector(
    selectCanvasSlice,
    (canvas) => selectEntity(canvas, this.entityIdentifier) as T | undefined
  );

  /**
   * A redux selector that selects the entity's position from the canvas slice.
   */
  selectPosition = createSelector(this.selectState, (entity) => entity?.position);

  syncIsOnscreen = () => {
    const stageRect = this.manager.stage.getScaledStageRect();
    const isOnScreen = this.checkIntersection(stageRect);
    const prevIsOnScreen = this.$isOnScreen.get();
    this.$isOnScreen.set(isOnScreen);
    if (prevIsOnScreen !== isOnScreen) {
      this.log.trace(`Moved ${isOnScreen ? 'on-screen' : 'off-screen'}`);
    }
    this.syncVisibility();
  };

  syncIntersectsBbox = () => {
    const bboxRect = this.manager.stateApi.getBbox().rect;
    const intersectsBbox = this.checkIntersection(bboxRect);
    const prevIntersectsBbox = this.$intersectsBbox.get();
    this.$intersectsBbox.set(intersectsBbox);
    if (prevIntersectsBbox !== intersectsBbox) {
      this.log.trace(`Moved ${intersectsBbox ? 'into bbox' : 'out of bbox'}`);
    }
  };

  checkIntersection = (rect: Rect): boolean => {
    const entityRect = this.transformer.$pixelRect.get();
    const position = this.manager.stateApi.runSelector(this.selectPosition);
    if (!position) {
      return false;
    }
    const entityRectRelativeToStage = {
      x: entityRect.x + position.x,
      y: entityRect.y + position.y,
      width: entityRect.width,
      height: entityRect.height,
    };
    const intersection = getRectIntersection(rect, entityRectRelativeToStage);
    const doesIntersect = intersection.width > 0 && intersection.height > 0;
    return doesIntersect;
  };

  initialize = async () => {
    this.log.debug('Initializing module');

    /**
     * When the pixel rect changes, we need to sync the onscreen state of the parent entity.
     *
     * TODO(psyche): It'd be nice to set this listener in the constructor, but the transformer is only created in the
     * concrete classes, so we have to do this here. IIRC the reason for this awkwardness was to satisfy a circular
     * dependency between the transformer and concrete adapter classes
     */
    this.subscriptions.add(this.transformer.$pixelRect.listen(this.syncIsOnscreen));

    /**
     * When the pixel rect changes, we need to sync the bbox intersection state of the parent entity.
     *
     * TODO(psyche): It'd be nice to set this listener in the constructor, but the transformer is only created in the
     * concrete classes, so we have to do this here. IIRC the reason for this awkwardness was to satisfy a circular
     * dependency between the transformer and concrete adapter classes
     */
    this.subscriptions.add(this.transformer.$pixelRect.listen(this.syncIntersectsBbox));

    await this.sync(this.manager.stateApi.runSelector(this.selectState), undefined);
    this.transformer.initialize();
    await this.renderer.initialize();
    this.syncZIndices();
    this.syncVisibility();

    // Call the init callbacks.
    // TODO(psyche): Get rid of the cast - see note in type def for `InitCallback`.
    CanvasEntityAdapterBase.runInitCallbacks(this as CanvasEntityAdapter);
  };

  syncZIndices = () => {
    this.log.trace('Updating z-indices');
    let zIndex = 0;
    this.renderer.konva.objectGroup.zIndex(zIndex++);
    this.bufferRenderer.konva.group.zIndex(zIndex++);
    if (this.renderer.konva.compositing) {
      this.renderer.konva.compositing.group.zIndex(zIndex++);
    }
    this.transformer.konva.outlineRect.zIndex(zIndex++);
    this.transformer.konva.proxyRect.zIndex(zIndex++);
    this.transformer.konva.transformer.zIndex(zIndex++);
  };

  /**
   * Synchronizes the enabled state of the entity with the canvas.
   */
  syncIsEnabled = () => {
    this.log.trace('Updating visibility');
    this.$isDisabled.set(!this.state.isEnabled);
    this.syncVisibility();
    this.transformer.syncInteractionState();
  };

  /**
   * Synchronizes the entity's objects with the canvas.
   */
  syncObjects = async () => {
    this.$isEmpty.set(this.state.objects.length === 0);
    const didRender = await this.renderer.render();
    if (didRender) {
      // If the objects have changed, we need to recalculate the transformer's bounding box.
      this.transformer.requestRectCalculation();
      this.transformer.syncInteractionState();
    }
  };

  /**
   * Synchronizes the entity's position with the canvas.
   */
  syncPosition = () => {
    this.transformer.updatePosition();
  };

  /**
   * Synchronizes the entity's opacity with the canvas.
   */
  syncOpacity = () => {
    this.renderer.updateOpacity();
  };

  syncVisibility = rafThrottle(() => {
    /**
     * If the entity type is hidden, so should the entity be hidden.
     */
    if (this.manager.stateApi.runSelector(this.selectIsTypeHidden)) {
      this.setVisibility(false);
      return;
    }

    if (this.manager.stateApi.runSelector(selectIsolatedStagingPreview)) {
      /**
       * When staging w/ isolatedStagingPreview enabled, we only show raster layers.
       *
       * This allows the user to easily see how the new generation fits in with the rest of the canvas without the
       * other layer types getting in the way.
       */
      const isStaging = this.manager.stateApi.runSelector(selectIsStaging);
      const isRasterLayer = isRasterLayerEntityIdentifier(this.entityIdentifier);
      if (isStaging && !isRasterLayer) {
        this.setVisibility(false);
        return;
      }
    }

    if (this.manager.stateApi.runSelector(selectIsolatedLayerPreview)) {
      /**
       * Handle isolated preview modes - if another entity is filtering, transforming, or select-object-ing, we may need
       * to hide this entity.
       */
      const filteringAdapter = this.manager.stateApi.$filteringAdapter.get();
      if (filteringAdapter && filteringAdapter !== this) {
        this.setVisibility(false);
        return;
      }

      const transformingAdapter = this.manager.stateApi.$transformingAdapter.get();
      if (
        transformingAdapter &&
        transformingAdapter !== this &&
        // Silent transforms should be transparent to the user, so we don't need to hide the entity.
        !transformingAdapter.transformer.$silentTransform.get()
      ) {
        this.setVisibility(false);
        return;
      }

      const segmentingAdapter = this.manager.stateApi.$segmentingAdapter.get();
      if (segmentingAdapter && segmentingAdapter !== this) {
        this.setVisibility(false);
        return;
      }
    }

    /**
     * Disabled entities should be hidden.
     */
    if (this.$isDisabled.get()) {
      this.setVisibility(false);
      return;
    }

    /**
     * When the entity is offscreen and not selected, we should hide it. If it is selected and offscreen, it still needs
     * to be visible so the user can interact with it.
     */
    if (!this.$isOnScreen.get() && !this.manager.stateApi.getIsSelected(this.entityIdentifier.id)) {
      this.setVisibility(false);
      return;
    }

    this.setVisibility(true);
  });

  setVisibility = (isVisible: boolean) => {
    const isLayerVisible = this.konva.layer.visible();

    if (isLayerVisible === isVisible) {
      // No change
      return;
    }
    this.log.trace(isVisible ? 'Showing' : 'Hiding');
    this.konva.layer.visible(isVisible);
    if (isVisible) {
      /**
       * When a layer is created and initially not visible, its compositing rect won't be set up properly. Then, when
       * we show it in this method, it the layer will not render as it should.
       *
       * For example, if an inpaint mask is created via select-object while the isolated layer preview feature is
       * enabled, it will be hidden on its first render, and the compositing rect will not be sized/positioned/filled.
       * When next show the layer, the its underlying objects will be rendered directly, without the compositing rect
       * providing the correct fill.
       *
       * The simplest way to ensure this doesn't happen is to always update the compositing rect when showing the layer.
       */
      this.renderer.updateCompositingRectSize();
      this.renderer.updateCompositingRectPosition();
      this.renderer.updateCompositingRectFill();
    }
    this.renderer.syncKonvaCache();
  };

  /**
   * Synchronizes the entity's locked state with the canvas.
   */
  syncIsLocked = () => {
    // The only thing we need to do is update the transformer's interaction state. For tool interactions, like drawing
    // shapes, we defer to the CanvasToolModule to handle the locked state.
    this.$isLocked.set(this.state.isLocked);
    this.transformer.syncInteractionState();
  };

  /**
   * Gets a hash of the entity's state, as provided by `getHashableState`. If `extra` is provided, it will be included in
   * the hash.
   */
  hash = (extra?: Jsonifiable): string => {
    const arg = {
      state: this.getHashableState(),
      extra,
    };
    return stableHash(arg);
  };

  cropToBbox = async (): Promise<ImageDTO> => {
    const { rect } = this.manager.stateApi.getBbox();
    const rasterizeResult = await withResultAsync(() =>
      this.renderer.rasterize({ rect, replaceObjects: true, attrs: { opacity: 1, filters: [] } })
    );
    if (rasterizeResult.isErr()) {
      toast({ status: 'error', title: 'Failed to crop to bbox' });
      throw rasterizeResult.error;
    }
    return rasterizeResult.value;
  };

  destroy = (): void => {
    this.log.debug('Destroying module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.subscriptions.clear();
    this.renderer.destroy();
    if (this.transformer.$isTransforming.get()) {
      this.transformer.stopTransform();
    }
    this.transformer.destroy();
    if (this.filterer) {
      if (this.filterer.$isFiltering.get()) {
        this.filterer.cancel();
      }
      this.filterer?.destroy();
    }
    if (this.segmentAnything) {
      if (this.segmentAnything.$isSegmenting.get()) {
        this.segmentAnything.cancel();
      }
      this.segmentAnything.destroy();
    }
    this.konva.layer.destroy();
    this.manager.deleteAdapter(this.entityIdentifier);
  };

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
      entityIdentifier: this.entityIdentifier,
      state: deepClone(this.state),
      transformer: this.transformer.repr(),
      renderer: this.renderer.repr(),
      bufferRenderer: this.bufferRenderer.repr(),
      segmentAnything: this.segmentAnything?.repr() ?? null,
      filterer: this.filterer?.repr() ?? null,
      hasCache: this.$canvasCache.get() !== null,
      isLocked: this.$isLocked.get(),
      isDisabled: this.$isDisabled.get(),
      isEntityTypeHidden: this.$isEntityTypeHidden.get(),
      isEmpty: this.$isEmpty.get(),
      isOnScreen: this.$isOnScreen.get(),
      intersectsBbox: this.$intersectsBbox.get(),
      konva: getKonvaNodeDebugAttrs(this.konva.layer),
    };
  };
}
