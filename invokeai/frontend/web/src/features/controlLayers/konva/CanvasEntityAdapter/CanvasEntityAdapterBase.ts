import type { SerializableObject } from 'common/types';
import { deepClone } from 'common/util/deepClone';
import type { CanvasEntityObjectRenderer } from 'features/controlLayers/konva/CanvasEntityObjectRenderer';
import type { CanvasEntityTransformer } from 'features/controlLayers/konva/CanvasEntityTransformer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import { selectEntity } from 'features/controlLayers/store/selectors';
import type { CanvasEntityIdentifier, CanvasRenderableEntityState, Rect } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { Logger } from 'roarr';
import stableHash from 'stable-hash';
import { assert } from 'tsafe';

export abstract class CanvasEntityAdapterBase<
  T extends CanvasRenderableEntityState = CanvasRenderableEntityState,
> extends CanvasModuleBase {
  readonly type: string;
  readonly id: string;
  readonly path: string[];
  readonly manager: CanvasManager;
  readonly parent: CanvasManager;
  readonly log: Logger;

  readonly entityIdentifier: CanvasEntityIdentifier<T['type']>;

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
   * The transformer for this entity adapter.
   */
  abstract transformer: CanvasEntityTransformer;

  /**
   * The renderer for this entity adapter.
   */
  abstract renderer: CanvasEntityObjectRenderer;

  /**
   * The entity's state.
   */
  state: T;

  /**
   * A set of subscriptions to stores.
   */
  subscriptions = new Set<() => void>();

  constructor(entityIdentifier: CanvasEntityIdentifier<T['type']>, manager: CanvasManager, adapterType: string) {
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

    // On creation, we need to get the latest snapshot of the entity's state from the store.
    const initialState = this.getSnapshot();
    assert(initialState !== undefined, 'Missing entity state on creation');
    this.state = initialState;
  }

  /**
   * Gets the latest snapshot of the entity's state from the store. If the entity does not exist, returns undefined.
   */
  getSnapshot = (): T | undefined => {
    return selectEntity(this.manager.stateApi.getCanvasState(), this.entityIdentifier) as T | undefined;
  };

  /**
   * Syncs the entity state with the canvas. This includes rendering the entity's objects, handling visibility,
   * positioning, opacity, locked state, and any other properties.
   *
   * Implementations should be minimal and should only update the canvas if the state has changed. However, if `force`
   * is true, the entity should be updated regardless of whether the state has changed.
   *
   * If the entity cannot be rendered, it should be destroyed.
   */
  abstract sync: (force?: boolean) => void;

  /**
   * Gets the canvas element for the entity. If `rect` is provided, the canvas will be clipped to that rectangle.
   */
  abstract getCanvas: (rect?: Rect) => HTMLCanvasElement;

  /**
   * Gets a hashable representation of the entity's state.
   */
  abstract getHashableState: () => SerializableObject;

  /**
   * Synchronizes the enabled state of the entity with the canvas.
   */
  syncIsEnabled = () => {
    this.log.trace('Updating visibility');
    this.konva.layer.visible(this.state.isEnabled);
    this.renderer.syncCache(this.state.isEnabled);
  };

  /**
   * Synchronizes the entity's objects with the canvas.
   */
  syncObjects = () => {
    this.renderer.render().then((didRender) => {
      if (didRender) {
        // If the objects have changed, we need to recalculate the transformer's bounding box.
        this.transformer.requestRectCalculation();
      }
    });
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
    this.transformer.destroy();
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
