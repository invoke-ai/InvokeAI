import type { SerializableObject } from 'common/types';
import { deepClone } from 'common/util/deepClone';
import { CanvasEntityRenderer } from 'features/controlLayers/konva/CanvasEntityRenderer';
import { CanvasEntityTransformer } from 'features/controlLayers/konva/CanvasEntityTransformer';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import type { CanvasEntityIdentifier, CanvasInpaintMaskState, Rect } from 'features/controlLayers/store/types';
import { getEntityIdentifier } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { GroupConfig } from 'konva/lib/Group';
import { omit } from 'lodash-es';
import type { Logger } from 'roarr';
import stableHash from 'stable-hash';
import { assert } from 'tsafe';

export class CanvasInpaintMaskAdapter extends CanvasModuleBase {
  readonly type = 'inpaint_mask_adapter';
  readonly id: string;
  readonly path: string[];
  readonly parent: CanvasManager;
  readonly manager: CanvasManager;
  readonly log: Logger;

  entityIdentifier: CanvasEntityIdentifier<'inpaint_mask'>;

  /**
   * The last known state of the entity.
   */
  private _state: CanvasInpaintMaskState | null = null;

  transformer: CanvasEntityTransformer;
  renderer: CanvasEntityRenderer;

  konva: {
    layer: Konva.Layer;
  };

  constructor(entityIdentifier: CanvasEntityIdentifier<'inpaint_mask'>, manager: CanvasManager) {
    super();
    this.id = entityIdentifier.id;
    this.entityIdentifier = entityIdentifier;
    this.parent = manager;
    this.manager = manager;
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

  get state(): CanvasInpaintMaskState {
    if (this._state) {
      return this._state;
    }
    const state = this.manager.stateApi.getInpaintMasksState().entities.find((layer) => layer.id === this.id);
    assert(state, `State not found for ${this.id}`);
    return state;
  }

  set state(state: CanvasInpaintMaskState) {
    this._state = state;
  }

  /**
   * Get this entity's entity identifier
   */
  getEntityIdentifier = (): CanvasEntityIdentifier => {
    return getEntityIdentifier(this.state);
  };

  update = async (state: CanvasInpaintMaskState) => {
    const prevState = this.state;
    this.state = state;

    if (prevState && prevState === state) {
      this.log.trace('State unchanged, skipping update');
      return;
    }

    if (!prevState || state.isEnabled !== prevState.isEnabled) {
      this.log.trace('Updating visibility');
      this.konva.layer.visible(state.isEnabled);
      this.renderer.syncCache(state.isEnabled);
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
    if (!prevState || state.isLocked !== prevState.isLocked) {
      this.transformer.syncInteractionState();
    }
    if (!prevState || state.fill !== prevState.fill) {
      this.renderer.updateCompositingRectFill(state.fill);
    }

    if (!prevState) {
      this.renderer.updateCompositingRectSize();
    }

    if (!prevState) {
      this.transformer.updateBbox();
    }
  };

  getHashableState = (): SerializableObject => {
    const keysToOmit: (keyof CanvasInpaintMaskState)[] = ['fill', 'name', 'opacity'];
    return omit(this.state, keysToOmit);
  };

  hash = (extra?: SerializableObject): string => {
    const arg = {
      state: this.getHashableState(),
      extra,
    };
    return stableHash(arg);
  };

  getCanvas = (rect?: Rect): HTMLCanvasElement => {
    // The opacity may have been changed in response to user selecting a different entity category, and the mask regions
    // should be fully opaque - set opacity to 1 before rendering the canvas
    const attrs: GroupConfig = { opacity: 1 };
    const canvas = this.renderer.getCanvas(rect, attrs);
    return canvas;
  };

  isInteractable = (): boolean => {
    return this.state.isEnabled && !this.state.isLocked;
  };

  destroy = () => {
    this.log.debug('Destroying module');
    this.transformer.destroy();
    this.renderer.destroy();
    this.konva.layer.destroy();
  };

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
      state: deepClone(this.state),
    };
  };
}
