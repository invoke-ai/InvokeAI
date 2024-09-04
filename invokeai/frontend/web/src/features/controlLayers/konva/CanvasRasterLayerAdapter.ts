import type { SerializableObject } from 'common/types';
import { deepClone } from 'common/util/deepClone';
import { CanvasEntityRenderer } from 'features/controlLayers/konva/CanvasEntityRenderer';
import { CanvasEntityTransformer } from 'features/controlLayers/konva/CanvasEntityTransformer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import type { CanvasEntityIdentifier, CanvasRasterLayerState, Rect } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { GroupConfig } from 'konva/lib/Group';
import { omit } from 'lodash-es';
import type { Logger } from 'roarr';
import stableHash from 'stable-hash';
import { assert } from 'tsafe';

export class CanvasRasterLayerAdapter extends CanvasModuleBase {
  readonly type = 'raster_layer_adapter';
  readonly id: string;
  readonly path: string[];
  readonly manager: CanvasManager;
  readonly parent: CanvasManager;
  readonly log: Logger;

  entityIdentifier: CanvasEntityIdentifier<'raster_layer'>;

  /**
   * The last known state of the entity.
   */
  private _state: CanvasRasterLayerState | null = null;

  /**
   * The Konva nodes that make up the entity layer:
   * - A layer to hold the everything
   *
   * Note that the transformer and object renderer have their own Konva nodes, but they are not stored here.
   */
  konva: {
    layer: Konva.Layer;
  };

  /**
   * The transformer for this entity layer.
   */
  transformer: CanvasEntityTransformer;

  /**
   * The renderer for this entity layer.
   */
  renderer: CanvasEntityRenderer;

  constructor(entityIdentifier: CanvasEntityIdentifier<'raster_layer'>, manager: CanvasManager) {
    super();
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

    this.renderer = new CanvasEntityRenderer(this);
    this.transformer = new CanvasEntityTransformer(this);
  }

  get state(): CanvasRasterLayerState {
    if (this._state) {
      return this._state;
    }
    const state = this.manager.stateApi.getRasterLayersState().entities.find((layer) => layer.id === this.id);
    assert(state, `State not found for ${this.id}`);
    return state;
  }

  set state(state: CanvasRasterLayerState) {
    const prevState = this._state;
    this._state = state;
    this.render(state, prevState);
  }

  private render = async (state: CanvasRasterLayerState, prevState: CanvasRasterLayerState | null): Promise<void> => {
    if (prevState && prevState === state) {
      this.log.trace('State unchanged, skipping update');
      return;
    }

    if (!prevState || state.isEnabled !== prevState.isEnabled) {
      this.log.trace('Updating visibility');
      this.konva.layer.visible(state.isEnabled);
      this.renderer.syncCache(state.isEnabled);
    }
    if (!prevState || state.isLocked !== prevState.isLocked) {
      this.transformer.syncInteractionState();
    }
    if (!prevState || state.objects !== prevState.objects) {
      const didRender = await this.renderer.render(this.state.objects);
      if (didRender) {
        this.transformer.requestRectCalculation();
      }
    }
    if (!prevState || state.position !== prevState.position) {
      this.transformer.updatePosition();
    }
    if (!prevState || state.opacity !== prevState.opacity) {
      this.renderer.updateOpacity(state.opacity);
    }
    if (!prevState) {
      // First render
      this.transformer.updateBbox();
    }
  };

  getCanvas = (rect?: Rect): HTMLCanvasElement => {
    this.log.trace({ rect }, 'Getting canvas');
    // The opacity may have been changed in response to user selecting a different entity category, so we must restore
    // the original opacity before rendering the canvas
    const attrs: GroupConfig = { opacity: this.state.opacity };
    const canvas = this.renderer.getCanvas(rect, attrs);
    return canvas;
  };

  isInteractable = (): boolean => {
    return this.state.isEnabled && !this.state.isLocked;
  };

  getHashableState = (): SerializableObject => {
    const keysToOmit: (keyof CanvasRasterLayerState)[] = ['name'];
    return omit(this.state, keysToOmit);
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
