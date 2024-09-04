import type { SerializableObject } from 'common/types';
import { deepClone } from 'common/util/deepClone';
import { CanvasEntityObjectRenderer } from 'features/controlLayers/konva/CanvasEntityObjectRenderer';
import { CanvasEntityTransformer } from 'features/controlLayers/konva/CanvasEntityTransformer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import type { CanvasEntityIdentifier, CanvasRenderableEntityState, Rect } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { Logger } from 'roarr';
import stableHash from 'stable-hash';

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
  transformer: CanvasEntityTransformer;

  /**
   * The renderer for this entity adapter.
   */
  renderer: CanvasEntityObjectRenderer;

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

    this.renderer = new CanvasEntityObjectRenderer(this);
    this.transformer = new CanvasEntityTransformer(this);
  }

  abstract get state(): T;

  abstract set state(state: T);

  abstract getCanvas: (rect?: Rect) => HTMLCanvasElement;

  abstract getHashableState: () => SerializableObject;

  isInteractable = (): boolean => {
    return this.state.isEnabled && !this.state.isLocked;
  };

  hash = (extra?: SerializableObject): string => {
    const arg = {
      state: this.getHashableState(),
      extra,
    };
    return stableHash(arg);
  };

  destroy = (): void => {
    this.log.debug('Destroying module');
    this.renderer.destroy();
    this.transformer.destroy();
    this.konva.layer.destroy();
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
