import { deepClone } from 'common/util/deepClone';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasObjectRenderer } from 'features/controlLayers/konva/CanvasObjectRenderer';
import { CanvasTransformer } from 'features/controlLayers/konva/CanvasTransformer';
import type {
  CanvasEntityIdentifier,
  CanvasLayerState,
  CanvasV2State,
  GetLoggingContext,
} from 'features/controlLayers/store/types';
import Konva from 'konva';
import { get } from 'lodash-es';
import type { Logger } from 'roarr';

export class CanvasLayerAdapter {
  id: string;
  type: CanvasLayerState['type'];
  manager: CanvasManager;
  log: Logger;
  getLoggingContext: GetLoggingContext;

  state: CanvasLayerState;

  konva: {
    layer: Konva.Layer;
  };
  transformer: CanvasTransformer;
  renderer: CanvasObjectRenderer;

  isFirstRender: boolean = true;

  constructor(state: CanvasLayerAdapter['state'], manager: CanvasLayerAdapter['manager']) {
    this.id = state.id;
    this.type = state.type;
    this.manager = manager;
    this.getLoggingContext = this.manager.buildGetLoggingContext(this);
    this.log = this.manager.buildLogger(this.getLoggingContext);
    this.log.debug({ state }, 'Creating layer');

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

    this.state = state;
  }

  /**
   * Get this entity's entity identifier
   */
  getEntityIdentifier = (): CanvasEntityIdentifier => {
    return { id: this.id, type: this.type };
  };

  destroy = (): void => {
    this.log.debug('Destroying layer');
    // We need to call the destroy method on all children so they can do their own cleanup.
    this.transformer.destroy();
    this.renderer.destroy();
    this.konva.layer.destroy();
  };

  update = async (arg?: {
    state: CanvasLayerAdapter['state'];
    toolState: CanvasV2State['tool'];
    isSelected: boolean;
  }) => {
    const state = get(arg, 'state', this.state);

    if (!this.isFirstRender && state === this.state) {
      this.log.trace('State unchanged, skipping update');
      return;
    }

    this.log.debug('Updating');
    const { position, objects, opacity, isEnabled } = state;

    if (this.isFirstRender || isEnabled !== this.state.isEnabled) {
      this.updateVisibility({ isEnabled });
    }
    if (this.isFirstRender || objects !== this.state.objects) {
      await this.updateObjects({ objects });
    }
    if (this.isFirstRender || position !== this.state.position) {
      this.transformer.updatePosition({ position });
    }
    if (this.isFirstRender || opacity !== this.state.opacity) {
      this.updateOpacity({ opacity });
    }
    // this.transformer.syncInteractionState();

    if (this.isFirstRender) {
      this.transformer.updateBbox();
    }

    this.state = state;
    this.isFirstRender = false;
  };

  updateVisibility = (arg?: { isEnabled: boolean }) => {
    this.log.trace('Updating visibility');
    const isEnabled = get(arg, 'isEnabled', this.state.isEnabled);
    this.konva.layer.visible(isEnabled);
  };

  updateObjects = async (arg?: { objects: CanvasLayerState['objects'] }) => {
    this.log.trace('Updating objects');

    const objects = get(arg, 'objects', this.state.objects);

    const didUpdate = await this.renderer.render(objects);

    if (didUpdate) {
      this.transformer.requestRectCalculation();
    }
  };

  updateOpacity = (arg?: { opacity: number }) => {
    this.log.trace('Updating opacity');
    const opacity = get(arg, 'opacity', this.state.opacity);
    this.renderer.konva.objectGroup.opacity(opacity);
  };

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      state: deepClone(this.state),
      transformer: this.transformer.repr(),
      renderer: this.renderer.repr(),
    };
  };

  logDebugInfo(msg = 'Debug info') {
    const info = {
      repr: this.repr(),
      interactionRectAttrs: {
        x: this.transformer.konva.proxyRect.x(),
        y: this.transformer.konva.proxyRect.y(),
        scaleX: this.transformer.konva.proxyRect.scaleX(),
        scaleY: this.transformer.konva.proxyRect.scaleY(),
        width: this.transformer.konva.proxyRect.width(),
        height: this.transformer.konva.proxyRect.height(),
        rotation: this.transformer.konva.proxyRect.rotation(),
      },
      objectGroupAttrs: {
        x: this.renderer.konva.objectGroup.x(),
        y: this.renderer.konva.objectGroup.y(),
        scaleX: this.renderer.konva.objectGroup.scaleX(),
        scaleY: this.renderer.konva.objectGroup.scaleY(),
        width: this.renderer.konva.objectGroup.width(),
        height: this.renderer.konva.objectGroup.height(),
        rotation: this.renderer.konva.objectGroup.rotation(),
        offsetX: this.renderer.konva.objectGroup.offsetX(),
        offsetY: this.renderer.konva.objectGroup.offsetY(),
      },
    };
    this.log.trace(info, msg);
  }
}
