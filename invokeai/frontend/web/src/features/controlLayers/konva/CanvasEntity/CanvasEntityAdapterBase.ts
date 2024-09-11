import { createSelector } from '@reduxjs/toolkit';
import type { SerializableObject } from 'common/types';
import { deepClone } from 'common/util/deepClone';
import type { CanvasEntityBufferObjectRenderer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityBufferObjectRenderer';
import type { CanvasEntityFilterer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityFilterer';
import type { CanvasEntityObjectRenderer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityObjectRenderer';
import type { CanvasEntityTransformer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityTransformer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { getIsHiddenSelector, selectCanvasSlice, selectEntity } from 'features/controlLayers/store/selectors';
import type { CanvasEntityIdentifier, CanvasRenderableEntityState, Rect } from 'features/controlLayers/store/types';
import Konva from 'konva';
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

    // When the hidden flag is updated, we need to update the entity's visibility and transformer interaction state,
    // which will show/hide the entity's selection outline
    this.subscriptions.add(
      this.manager.stateApi.createStoreSubscription(getIsHiddenSelector(this.entityIdentifier.type), () => {
        this.syncOpacity();
        this.transformer.syncInteractionState();
      })
    );
  }

  /**
   * A redux selector that selects the entity's state from the canvas slice.
   */
  selectState = createSelector(
    selectCanvasSlice,
    (canvas) => selectEntity(canvas, this.entityIdentifier) as T | undefined
  );

  initialize = async () => {
    this.log.debug('Initializing module');
    await this.sync(this.manager.stateApi.runSelector(this.selectState), undefined);
    this.transformer.initialize();
    await this.renderer.initialize();
    this.syncZIndices();
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
  };

  /**
   * Synchronizes the entity's objects with the canvas.
   */
  syncObjects = async () => {
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

  /**
   * Synchronizes the entity's locked state with the canvas.
   */
  syncIsLocked = () => {
    // The only thing we need to do is update the transformer's interaction state. For tool interactions, like drawing
    // shapes, we defer to the CanvasToolModule to handle the locked state.
    this.transformer.syncInteractionState();
  };

  /**
   * Checks if the entity is interactable. An entity is interactable if it is enabled, not locked, and its type is not
   * hidden.
   */
  getIsInteractable = (): boolean => {
    return this.state.isEnabled && !this.state.isLocked && !this.manager.stateApi.getIsTypeHidden(this.state.type);
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
      state: deepClone(this.state),
      transformer: this.transformer.repr(),
      renderer: this.renderer.repr(),
    };
  };
}
