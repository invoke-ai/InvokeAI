import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasObjectRenderer } from 'features/controlLayers/konva/CanvasObjectRenderer';
import { CanvasTransformer } from 'features/controlLayers/konva/CanvasTransformer';
import type {
  CanvasInpaintMaskState,
  CanvasRegionalGuidanceState,
  CanvasV2State,
  GetLoggingContext,
} from 'features/controlLayers/store/types';
import Konva from 'konva';
import { get } from 'lodash-es';
import type { Logger } from 'roarr';

export class CanvasMaskAdapter {
  id: string;
  type: CanvasInpaintMaskState['type'] | CanvasRegionalGuidanceState['type'];
  manager: CanvasManager;
  log: Logger;
  getLoggingContext: GetLoggingContext;

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
    this.type = state.type;
    this.manager = manager;
    this.getLoggingContext = this.manager.buildGetLoggingContext(this);
    this.log = this.manager.buildLogger(this.getLoggingContext);
    this.log.debug({ state }, 'Creating mask');

    this.konva = {
      layer: new Konva.Layer({
        id: this.id,
        name: `${this.type}:layer`,
        listening: false,
        imageSmoothingEnabled: false,
      }),
    };

    this.renderer = new CanvasObjectRenderer(this);
    this.transformer = new CanvasTransformer(this);

    this.state = state;
    this.maskOpacity = this.manager.stateApi.getMaskOpacity();
  }

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
}
