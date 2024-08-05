import { getStore } from 'app/store/nanostores/store';
import { deepClone } from 'common/util/deepClone';
import { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasObjectRenderer } from 'features/controlLayers/konva/CanvasObjectRenderer';
import { CanvasTransformer } from 'features/controlLayers/konva/CanvasTransformer';
import { getEmptyRect, konvaNodeToBlob, previewBlob } from 'features/controlLayers/konva/util';
import { layerRasterized } from 'features/controlLayers/store/canvasV2Slice';
import type {
  CanvasLayerState,
  CanvasV2State,
  Coordinate,
  GetLoggingContext,
  Rect,
} from 'features/controlLayers/store/types';
import { imageDTOToImageObject } from 'features/controlLayers/store/types';
import Konva from 'konva';
import { debounce, get } from 'lodash-es';
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
  isPendingBboxCalculation: boolean = false;

  rect: Rect = getEmptyRect();
  bbox: Rect = getEmptyRect();

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
      await this.updatePosition({ position });
    }
    if (this.isFirstRender || opacity !== this.state.opacity) {
      await this.updateOpacity({ opacity });
    }
    if (this.isFirstRender || isEnabled !== this.state.isEnabled) {
      await this.updateVisibility({ isEnabled });
    }
    // this.transformer.syncInteractionState();

    if (this.isFirstRender) {
      await this.updateBbox();
    }

    this.state = state;
    this.isFirstRender = false;
  };

  updateVisibility = (arg?: { isEnabled: boolean }) => {
    this.log.trace('Updating visibility');
    const isEnabled = get(arg, 'isEnabled', this.state.isEnabled);
    this.konva.layer.visible(isEnabled && this.renderer.hasObjects());
  };

  updatePosition = (arg?: { position: Coordinate }) => {
    this.log.trace('Updating position');
    const position = get(arg, 'position', this.state.position);

    this.konva.objectGroup.setAttrs({
      x: position.x + this.bbox.x,
      y: position.y + this.bbox.y,
      offsetX: this.bbox.x,
      offsetY: this.bbox.y,
    });

    this.transformer.update(position, this.bbox);
  };

  updateObjects = async (arg?: { objects: CanvasLayerState['objects'] }) => {
    this.log.trace('Updating objects');

    const objects = get(arg, 'objects', this.state.objects);

    const didUpdate = await this.renderer.render(objects);

    if (didUpdate) {
      this.calculateBbox();
    }

    this.isFirstRender = false;
  };

  updateOpacity = (arg?: { opacity: number }) => {
    this.log.trace('Updating opacity');
    const opacity = get(arg, 'opacity', this.state.opacity);
    this.konva.objectGroup.opacity(opacity);
  };

  updateBbox = () => {
    this.log.trace('Updating bbox');

    if (this.isPendingBboxCalculation) {
      return;
    }

    // If the bbox has no width or height, that means the layer is fully transparent. This can happen if it is only
    // eraser lines, fully clipped brush lines or if it has been fully erased.
    if (this.bbox.width === 0 || this.bbox.height === 0) {
      // We shouldn't reset on the first render - the bbox will be calculated on the next render
      if (!this.isFirstRender && !this.renderer.hasObjects()) {
        // The layer is fully transparent but has objects - reset it
        this.manager.stateApi.onEntityReset({ id: this.id }, 'layer');
      }
      this.transformer.syncInteractionState();
      return;
    }

    this.transformer.syncInteractionState();
    this.transformer.update(this.state.position, this.bbox);
    this.konva.objectGroup.setAttrs({
      x: this.state.position.x + this.bbox.x,
      y: this.state.position.y + this.bbox.y,
      offsetX: this.bbox.x,
      offsetY: this.bbox.y,
    });
  };

  resetScale = () => {
    const attrs = {
      scaleX: 1,
      scaleY: 1,
      rotation: 0,
    };
    this.konva.objectGroup.setAttrs(attrs);
    this.transformer.konva.bboxOutline.setAttrs(attrs);
    this.transformer.konva.proxyRect.setAttrs(attrs);
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
    const { dispatch } = getStore();
    const imageObject = imageDTOToImageObject(imageDTO);
    await this.renderer.renderObject(imageObject, true);
    this.resetScale();
    dispatch(layerRasterized({ id: this.id, imageObject, position: { x: Math.round(rect.x), y: Math.round(rect.y) } }));
  };

  calculateBbox = debounce(() => {
    this.log.debug('Calculating bbox');

    this.isPendingBboxCalculation = true;

    if (!this.renderer.hasObjects()) {
      this.log.trace('No objects, resetting bbox');
      this.rect = getEmptyRect();
      this.bbox = getEmptyRect();
      this.isPendingBboxCalculation = false;
      this.updateBbox();
      return;
    }

    const rect = this.konva.objectGroup.getClientRect({ skipTransform: true });

    if (!this.renderer.needsPixelBbox()) {
      this.rect = deepClone(rect);
      this.bbox = deepClone(rect);
      this.isPendingBboxCalculation = false;
      this.log.trace({ bbox: this.bbox, rect: this.rect }, 'Got bbox from client rect');
      this.updateBbox();
      return;
    }

    // We have eraser strokes - we must calculate the bbox using pixel data

    const clone = this.konva.objectGroup.clone();
    const canvas = clone.toCanvas();
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return;
    }
    const imageData = ctx.getImageData(0, 0, rect.width, rect.height);
    this.manager.requestBbox(
      { buffer: imageData.data.buffer, width: imageData.width, height: imageData.height },
      (extents) => {
        if (extents) {
          const { minX, minY, maxX, maxY } = extents;
          this.rect = deepClone(rect);
          this.bbox = {
            x: rect.x + minX,
            y: rect.y + minY,
            width: maxX - minX,
            height: maxY - minY,
          };
        } else {
          this.bbox = getEmptyRect();
          this.rect = getEmptyRect();
        }
        this.isPendingBboxCalculation = false;
        this.log.trace({ bbox: this.bbox, rect: this.rect, extents }, `Got bbox from worker`);
        this.updateBbox();
        clone.destroy();
      }
    );
  }, CanvasManager.BBOX_DEBOUNCE_MS);

  repr = () => {
    return {
      id: this.id,
      type: CanvasLayer.TYPE,
      state: deepClone(this.state),
      rect: deepClone(this.rect),
      bbox: deepClone(this.bbox),
      bboxNeedsUpdate: this.bboxNeedsUpdate,
      isPendingBboxCalculation: this.isPendingBboxCalculation,
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