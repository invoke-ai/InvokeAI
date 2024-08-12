import type { JSONObject } from 'common/types';
import { deepClone } from 'common/util/deepClone';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasObjectRenderer } from 'features/controlLayers/konva/CanvasObjectRenderer';
import { CanvasTransformer } from 'features/controlLayers/konva/CanvasTransformer';
import type {
  CanvasEntityIdentifier,
  CanvasInpaintMaskState,
  CanvasRegionalGuidanceState,
  CanvasV2State,
} from 'features/controlLayers/store/types';
import Konva from 'konva';
import { get } from 'lodash-es';
import type { Logger } from 'roarr';

export class CanvasMaskAdapter {
  readonly type = 'mask_adapter';

  id: string;
  path: string[];
  manager: CanvasManager;
  log: Logger;

  state: CanvasInpaintMaskState | CanvasRegionalGuidanceState;
  maskOpacity: number;

  transformer: CanvasTransformer;
  renderer: CanvasObjectRenderer;

  isFirstRender: boolean = true;

  konva: {
    layer: Konva.Layer;
  };

  constructor(state: CanvasMaskAdapter['state'], manager: CanvasMaskAdapter['manager']) {
    this.id = state.id;
    this.manager = manager;
    this.path = this.manager.path.concat(this.id);
    this.log = this.manager.buildLogger(this.getLoggingContext);
    this.log.debug({ state }, 'Creating mask');
    this.state = state;

    this.konva = {
      layer: new Konva.Layer({
        // We need the ID on the layer to help with building the composite initial image
        // See `getCompositeLayerStageClone()`
        id: this.id,
        name: `${this.type}:layer`,
        listening: false,
        imageSmoothingEnabled: false,
      }),
    };

    this.renderer = new CanvasObjectRenderer(this);
    this.transformer = new CanvasTransformer(this);

    this.maskOpacity = this.manager.stateApi.getMaskOpacity();
  }

  /**
   * Get this entity's entity identifier
   */
  getEntityIdentifier = (): CanvasEntityIdentifier => {
    return { id: this.id, type: this.state.type };
  };

  destroy = (): void => {
    this.log.debug('Destroying mask');
    // We need to call the destroy method on all children so they can do their own cleanup.
    this.transformer.destroy();
    this.renderer.destroy();
    this.konva.layer.destroy();
  };

  update = async (arg?: {
    state: CanvasMaskAdapter['state'];
    toolState: CanvasV2State['tool'];
    isSelected: boolean;
  }) => {
    const state = get(arg, 'state', this.state);
    const maskOpacity = this.manager.stateApi.getMaskOpacity();

    if (
      !this.isFirstRender &&
      state === this.state &&
      state.fill === this.state.fill &&
      maskOpacity === this.maskOpacity
    ) {
      this.log.trace('State unchanged, skipping update');
      return;
    }

    this.log.debug('Updating');
    const { position, objects, isEnabled } = state;

    if (this.isFirstRender || objects !== this.state.objects) {
      await this.updateObjects({ objects });
    }
    if (this.isFirstRender || position !== this.state.position) {
      this.transformer.updatePosition({ position });
    }
    // if (this.isFirstRender || opacity !== this.state.opacity) {
    //   await this.updateOpacity({ opacity });
    // }
    if (this.isFirstRender || isEnabled !== this.state.isEnabled) {
      this.updateVisibility({ isEnabled });
    }

    if (this.isFirstRender || state.fill !== this.state.fill || maskOpacity !== this.maskOpacity) {
      this.renderer.updateCompositingRect(state.fill, maskOpacity);
      this.maskOpacity = maskOpacity;
    }
    // this.transformer.syncInteractionState();

    if (this.isFirstRender) {
      this.transformer.updateBbox();
    }

    this.state = state;
    this.isFirstRender = false;
  };

  updateObjects = async (arg?: { objects: CanvasInpaintMaskState['objects'] }) => {
    this.log.trace('Updating objects');

    const objects = get(arg, 'objects', this.state.objects);

    const didUpdate = await this.renderer.render(objects);

    if (didUpdate) {
      this.transformer.requestRectCalculation();
    }
  };

  updateVisibility = (arg?: { isEnabled: boolean }) => {
    this.log.trace('Updating visibility');
    const isEnabled = get(arg, 'isEnabled', this.state.isEnabled);
    this.konva.layer.visible(isEnabled);
  };

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      state: deepClone(this.state),
    };
  };

  getLoggingContext = (): JSONObject => {
    return { ...this.manager.getLoggingContext(), path: this.path.join('.') };
  };
}
