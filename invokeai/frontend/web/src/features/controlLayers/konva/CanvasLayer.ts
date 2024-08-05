import { deepClone } from 'common/util/deepClone';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasObjectRenderer } from 'features/controlLayers/konva/CanvasObjectRenderer';
import { CanvasTransformer } from 'features/controlLayers/konva/CanvasTransformer';
import { konvaNodeToBlob, previewBlob } from 'features/controlLayers/konva/util';
import type {
  CanvasLayerState,
  CanvasV2State,
  GetLoggingContext,
} from 'features/controlLayers/store/types';
import { imageDTOToImageObject } from 'features/controlLayers/store/types';
import Konva from 'konva';
import { get } from 'lodash-es';
import type { Logger } from 'roarr';
import { uploadImage } from 'services/api/endpoints/images';

export class CanvasLayer {
  static TYPE = 'layer' as const;
  static KONVA_LAYER_NAME = `${CanvasLayer.TYPE}_layer`;
  static KONVA_OBJECT_GROUP_NAME = `${CanvasLayer.TYPE}_object-group`;

  id: string;
  type = CanvasLayer.TYPE;
  manager: CanvasManager;
  log: Logger;
  getLoggingContext: GetLoggingContext;

  state: CanvasLayerState;

  konva: {
    layer: Konva.Layer;
    objectGroup: Konva.Group;
  };
  transformer: CanvasTransformer;
  renderer: CanvasObjectRenderer;

  isFirstRender: boolean = true;
  bboxNeedsUpdate: boolean = true;

  constructor(state: CanvasLayerState, manager: CanvasManager) {
    this.id = state.id;
    this.manager = manager;
    this.getLoggingContext = this.manager.buildGetLoggingContext(this);
    this.log = this.manager.buildLogger(this.getLoggingContext);
    this.log.debug({ state }, 'Creating layer');

    this.konva = {
      layer: new Konva.Layer({
        id: this.id,
        name: CanvasLayer.KONVA_LAYER_NAME,
        listening: false,
        imageSmoothingEnabled: false,
      }),
      objectGroup: new Konva.Group({ name: CanvasLayer.KONVA_OBJECT_GROUP_NAME, listening: false }),
    };

    this.transformer = new CanvasTransformer(this);
    this.renderer = new CanvasObjectRenderer(this);

    this.konva.layer.add(this.konva.objectGroup);
    this.konva.layer.add(...this.transformer.getNodes());

    this.state = state;
  }

  destroy = (): void => {
    this.log.debug('Destroying layer');
    // We need to call the destroy method on all children so they can do their own cleanup.
    this.transformer.destroy();
    this.renderer.destroy();
    this.konva.layer.destroy();
  };

  update = async (arg?: { state: CanvasLayerState; toolState: CanvasV2State['tool']; isSelected: boolean }) => {
    const state = get(arg, 'state', this.state);

    if (!this.isFirstRender && state === this.state) {
      this.log.trace('State unchanged, skipping update');
      return;
    }

    this.log.debug('Updating');
    const { position, objects, opacity, isEnabled } = state;

    if (this.isFirstRender || objects !== this.state.objects) {
      await this.updateObjects({ objects });
    }
    if (this.isFirstRender || position !== this.state.position) {
      await this.transformer.updatePosition({ position });
    }
    if (this.isFirstRender || opacity !== this.state.opacity) {
      await this.updateOpacity({ opacity });
    }
    if (this.isFirstRender || isEnabled !== this.state.isEnabled) {
      await this.updateVisibility({ isEnabled });
    }
    // this.transformer.syncInteractionState();

    if (this.isFirstRender) {
      await this.transformer.updateBbox();
    }

    this.state = state;
    this.isFirstRender = false;
  };

  updateVisibility = (arg?: { isEnabled: boolean }) => {
    this.log.trace('Updating visibility');
    const isEnabled = get(arg, 'isEnabled', this.state.isEnabled);
    this.konva.layer.visible(isEnabled && this.renderer.hasObjects());
  };

  updateObjects = async (arg?: { objects: CanvasLayerState['objects'] }) => {
    this.log.trace('Updating objects');

    const objects = get(arg, 'objects', this.state.objects);

    const didUpdate = await this.renderer.render(objects);

    if (didUpdate) {
      this.transformer.requestRectCalculation();
    }

    this.isFirstRender = false;
  };

  updateOpacity = (arg?: { opacity: number }) => {
    this.log.trace('Updating opacity');
    const opacity = get(arg, 'opacity', this.state.opacity);
    this.konva.objectGroup.opacity(opacity);
  };

  rasterize = async () => {
    this.log.debug('Rasterizing layer');

    const objectGroupClone = this.konva.objectGroup.clone();
    const interactionRectClone = this.transformer.konva.proxyRect.clone();
    const rect = interactionRectClone.getClientRect();
    const blob = await konvaNodeToBlob(objectGroupClone, rect);
    if (this.manager._isDebugging) {
      previewBlob(blob, 'Rasterized layer');
    }
    const imageDTO = await uploadImage(blob, `${this.id}_rasterized.png`, 'other', true);
    const imageObject = imageDTOToImageObject(imageDTO);
    await this.renderer.renderObject(imageObject, true);
    this.manager.stateApi.rasterizeEntity(
      { id: this.id, imageObject, position: { x: Math.round(rect.x), y: Math.round(rect.y) } },
      this.type
    );
  };

  repr = () => {
    return {
      id: this.id,
      type: CanvasLayer.TYPE,
      state: deepClone(this.state),
      bboxNeedsUpdate: this.bboxNeedsUpdate,
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
        x: this.konva.objectGroup.x(),
        y: this.konva.objectGroup.y(),
        scaleX: this.konva.objectGroup.scaleX(),
        scaleY: this.konva.objectGroup.scaleY(),
        width: this.konva.objectGroup.width(),
        height: this.konva.objectGroup.height(),
        rotation: this.konva.objectGroup.rotation(),
        offsetX: this.konva.objectGroup.offsetX(),
        offsetY: this.konva.objectGroup.offsetY(),
      },
    };
    this.log.trace(info, msg);
  }
}
