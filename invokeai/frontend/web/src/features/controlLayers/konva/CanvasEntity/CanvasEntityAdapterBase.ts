import type { Selector } from '@reduxjs/toolkit';
import { createSelector } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SerializableObject } from 'common/types';
import { deepClone } from 'common/util/deepClone';
import type { CanvasEntityBufferObjectRenderer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityBufferObjectRenderer';
import type { CanvasEntityFilterer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityFilterer';
import type { CanvasEntityObjectRenderer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityObjectRenderer';
import type { CanvasEntityTransformer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityTransformer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import {
  selectIsolatedFilteringPreview,
  selectIsolatedTransformingPreview,
} from 'features/controlLayers/store/canvasSettingsSlice';
import { buildEntityIsHiddenSelector, selectCanvasSlice, selectEntity } from 'features/controlLayers/store/selectors';
import type { CanvasEntityIdentifier, CanvasRenderableEntityState, Rect } from 'features/controlLayers/store/types';
import Konva from 'konva';
import { atom, computed } from 'nanostores';
import type { Logger } from 'roarr';
import stableHash from 'stable-hash';
import { assert } from 'tsafe';

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
   * Synchronizes the entity state with the canvas. This includes rendering the entity's objects, handling visibility,
   * positioning, opacity, locked state, and any other properties.
   *
   * Implementations should be minimal and should only update the canvas if the state has changed.
   *
   * If `state` is undefined, the entity was just deleted and the adapter should destroy itself.
   *
   * If `prevState` is undefined, this is the first time the entity is being synced.
   */
  abstract sync: (state: T | undefined, prevState: T | undefined) => void;

  /**
   * Gets the canvas element for the entity. If `rect` is provided, the canvas will be clipped to that rectangle.
   */
  abstract getCanvas: (rect?: Rect) => HTMLCanvasElement;

  /**
   * Gets a hashable representation of the entity's state.
   */
  abstract getHashableState: () => SerializableObject;

  private _selectIsHidden: Selector<RootState, boolean> | null = null;

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
  $isHidden = atom(false);
  /**
   * Whether this entity is empty. This is computed based on the entity's objects.
   */
  $isEmpty = atom(true);
  /**
   * Whether this entity is interactable. This is computed based on the entity's locked, disabled, and hidden states.
   */
  $isInteractable = computed([this.$isLocked, this.$isDisabled, this.$isHidden], (isLocked, isDisabled, isHidden) => {
    return !isLocked && !isDisabled && !isHidden;
  });

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

    /**
     * There are a number of reason we may need to show or hide a layer:
     * - The entity is enabled/disabled
     * - The entity type is hidden/shown
     * - Staging status changes and `isolatedStagingPreview` is enabled
     * - Global filtering status changes and `isolatedFilteringPreview` is enabled
     * - Global transforming status changes and `isolatedTransformingPreview` is enabled
     */
    this.subscriptions.add(this.manager.stateApi.createStoreSubscription(this.selectIsHidden, this.syncVisibility));
    this.subscriptions.add(
      this.manager.stateApi.createStoreSubscription(selectIsolatedFilteringPreview, this.syncVisibility)
    );
    this.subscriptions.add(this.manager.stateApi.$filteringAdapter.listen(this.syncVisibility));
    this.subscriptions.add(
      this.manager.stateApi.createStoreSubscription(selectIsolatedTransformingPreview, this.syncVisibility)
    );
    this.subscriptions.add(this.manager.stateApi.$transformingAdapter.listen(this.syncVisibility));

    /**
     * The tool preview may need to be updated when the entity is locked or disabled. For example, when we disable the
     * entity, we should hide the tool preview & change the cursor.
     */
    this.subscriptions.add(this.$isInteractable.subscribe(this.manager.tool.render));
  }

  /**
   * A redux selector that selects the entity's state from the canvas slice.
   */
  selectState = createSelector(
    selectCanvasSlice,
    (canvas) => selectEntity(canvas, this.entityIdentifier) as T | undefined
  );

  // This must be a getter because the selector depends on the entityIdentifier, which is set in the constructor.
  get selectIsHidden() {
    if (!this._selectIsHidden) {
      this._selectIsHidden = buildEntityIsHiddenSelector(this.entityIdentifier);
    }
    return this._selectIsHidden;
  }

  initialize = async () => {
    this.log.debug('Initializing module');
    await this.sync(this.manager.stateApi.runSelector(this.selectState), undefined);
    this.transformer.initialize();
    await this.renderer.initialize();
    this.syncZIndices();
    this.syncVisibility();
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
    this.konva.layer.visible(this.state.isEnabled);
    this.renderer.syncCache(this.state.isEnabled);
    this.transformer.syncInteractionState();
    this.$isDisabled.set(!this.state.isEnabled);
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

  syncVisibility = () => {
    let isHidden = this.manager.stateApi.runSelector(this.selectIsHidden);

    // Handle isolated preview modes - if another entity is filtering or transforming, we may need to hide this entity.
    if (this.manager.stateApi.runSelector(selectIsolatedFilteringPreview)) {
      const filteringEntityIdentifier = this.manager.stateApi.$filteringAdapter.get()?.entityIdentifier;
      if (filteringEntityIdentifier && filteringEntityIdentifier.id !== this.id) {
        isHidden = true;
      }
    }

    if (this.manager.stateApi.runSelector(selectIsolatedTransformingPreview)) {
      const transformingEntityIdentifier = this.manager.stateApi.$transformingAdapter.get()?.entityIdentifier;
      if (transformingEntityIdentifier && transformingEntityIdentifier.id !== this.id) {
        isHidden = true;
      }
    }

    this.$isHidden.set(isHidden);
    this.konva.layer.visible(!isHidden);
  };

  /**
   * Synchronizes the entity's locked state with the canvas.
   */
  syncIsLocked = () => {
    // The only thing we need to do is update the transformer's interaction state. For tool interactions, like drawing
    // shapes, we defer to the CanvasToolModule to handle the locked state.
    this.transformer.syncInteractionState();
    this.$isLocked.set(this.state.isLocked);
  };

  /**
   * Gets a hash of the entity's state, as provided by `getHashableState`. If `extra` is provided, it will be included in
   * the hash.
   */
  hash = (extra?: SerializableObject): string => {
    const arg = {
      state: this.getHashableState(),
      extra,
    };
    return stableHash(arg);
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
    if (this.filterer?.$isFiltering.get()) {
      this.filterer.cancel();
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
      filterer: this.filterer?.repr(),
    };
  };
}
