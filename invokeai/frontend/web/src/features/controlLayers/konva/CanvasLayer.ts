import { getStore } from 'app/store/nanostores/store';
import { deepClone } from 'common/util/deepClone';
import { CanvasBrushLine } from 'features/controlLayers/konva/CanvasBrushLine';
import { CanvasEntity } from 'features/controlLayers/konva/CanvasEntity';
import { CanvasEraserLine } from 'features/controlLayers/konva/CanvasEraserLine';
import { CanvasImage } from 'features/controlLayers/konva/CanvasImage';
import { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasRect } from 'features/controlLayers/konva/CanvasRect';
import { CanvasTransformer } from 'features/controlLayers/konva/CanvasTransformer';
import { getPrefixedId, konvaNodeToBlob, mapId, previewBlob } from 'features/controlLayers/konva/util';
import { layerRasterized } from 'features/controlLayers/store/canvasV2Slice';
import {
  type BrushLine,
  type CanvasV2State,
  type Coordinate,
  type EraserLine,
  imageDTOToImageObject,
  type LayerEntity,
  type Rect,
  type RectShape,
} from 'features/controlLayers/store/types';
import Konva from 'konva';
import { debounce, get } from 'lodash-es';
import { uploadImage } from 'services/api/endpoints/images';
import { assert } from 'tsafe';

export class CanvasLayer extends CanvasEntity {
  static TYPE = 'layer';
  static LAYER_NAME = `${CanvasLayer.TYPE}_layer`;
  static TRANSFORMER_NAME = `${CanvasLayer.TYPE}_transformer`;
  static INTERACTION_RECT_NAME = `${CanvasLayer.TYPE}_interaction-rect`;
  static GROUP_NAME = `${CanvasLayer.TYPE}_group`;
  static OBJECT_GROUP_NAME = `${CanvasLayer.TYPE}_object-group`;
  static BBOX_NAME = `${CanvasLayer.TYPE}_bbox`;

  drawingBuffer: BrushLine | EraserLine | RectShape | null;
  state: LayerEntity;

  konva: {
    layer: Konva.Layer;
    bbox: Konva.Rect;
    objectGroup: Konva.Group;
    interactionRect: Konva.Rect;
  };
  objects: Map<string, CanvasBrushLine | CanvasEraserLine | CanvasRect | CanvasImage>;
  transformer: CanvasTransformer;

  bboxNeedsUpdate: boolean;
  isFirstRender: boolean;
  isTransforming: boolean;
  isPendingBboxCalculation: boolean;

  rect: Rect;
  bbox: Rect;

  constructor(state: LayerEntity, manager: CanvasManager) {
    super(state.id, manager);
    this.log.debug({ state }, 'Creating layer');

    this.konva = {
      layer: new Konva.Layer({ id: this.id, name: CanvasLayer.LAYER_NAME, listening: false }),
      bbox: new Konva.Rect({
        listening: false,
        draggable: false,
        name: CanvasLayer.BBOX_NAME,
        stroke: 'hsl(200deg 76% 59%)', // invokeBlue.400
        perfectDrawEnabled: false,
        strokeHitEnabled: false,
      }),
      objectGroup: new Konva.Group({ name: CanvasLayer.OBJECT_GROUP_NAME, listening: false }),
      interactionRect: new Konva.Rect({
        name: CanvasLayer.INTERACTION_RECT_NAME,
        listening: false,
        draggable: true,
        // fill: 'rgba(255,0,0,0.5)',
      }),
    };

    this.transformer = new CanvasTransformer(this);

    this.konva.layer.add(this.konva.objectGroup);
    this.konva.layer.add(this.transformer.konva.transformer);
    this.konva.layer.add(this.konva.interactionRect);
    this.konva.layer.add(this.konva.bbox);

    this.konva.interactionRect.on('dragmove', () => {
      // Snap the interaction rect to the nearest pixel
      this.konva.interactionRect.x(Math.round(this.konva.interactionRect.x()));
      this.konva.interactionRect.y(Math.round(this.konva.interactionRect.y()));

      // The bbox should be updated to reflect the new position of the interaction rect, taking into account its padding
      // and border
      this.konva.bbox.setAttrs({
        x: this.konva.interactionRect.x() - this.manager.getScaledBboxPadding(),
        y: this.konva.interactionRect.y() - this.manager.getScaledBboxPadding(),
      });

      // The object group is translated by the difference between the interaction rect's new and old positions (which is
      // stored as this.bbox)
      this.konva.objectGroup.setAttrs({
        x: this.konva.interactionRect.x(),
        y: this.konva.interactionRect.y(),
      });
    });
    this.konva.interactionRect.on('dragend', () => {
      if (this.isTransforming) {
        // When the user cancels the transformation, we need to reset the layer, so we should not update the layer's
        // positition while we are transforming - bail out early.
        return;
      }

      const position = {
        x: this.konva.interactionRect.x() - this.bbox.x,
        y: this.konva.interactionRect.y() - this.bbox.y,
      };

      this.log.trace({ position }, 'Position changed');
      this.manager.stateApi.onPosChanged({ id: this.id, position }, 'layer');
    });

    this.objects = new Map();
    this.drawingBuffer = null;
    this.state = state;
    this.rect = this.getDefaultRect();
    this.bbox = this.getDefaultRect();
    this.bboxNeedsUpdate = true;
    this.isTransforming = false;
    this.isFirstRender = true;
    this.isPendingBboxCalculation = false;
  }

  destroy = (): void => {
    this.log.debug('Destroying layer');
    this.konva.layer.destroy();
  };

  getDrawingBuffer = () => {
    return this.drawingBuffer;
  };

  setDrawingBuffer = async (obj: BrushLine | EraserLine | RectShape | null) => {
    if (obj) {
      this.drawingBuffer = obj;
      await this._renderObject(this.drawingBuffer, true);
    } else {
      this.drawingBuffer = null;
    }
  };

  finalizeDrawingBuffer = async () => {
    if (!this.drawingBuffer) {
      return;
    }
    const drawingBuffer = this.drawingBuffer;
    await this.setDrawingBuffer(null);

    // We need to give the objects a fresh ID else they will be considered the same object when they are re-rendered as
    // a non-buffer object, and we won't trigger things like bbox calculation

    if (drawingBuffer.type === 'brush_line') {
      drawingBuffer.id = getPrefixedId('brush_line');
      this.manager.stateApi.onBrushLineAdded({ id: this.id, brushLine: drawingBuffer }, 'layer');
    } else if (drawingBuffer.type === 'eraser_line') {
      drawingBuffer.id = getPrefixedId('brush_line');
      this.manager.stateApi.onEraserLineAdded({ id: this.id, eraserLine: drawingBuffer }, 'layer');
    } else if (drawingBuffer.type === 'rect_shape') {
      drawingBuffer.id = getPrefixedId('brush_line');
      this.manager.stateApi.onRectShapeAdded({ id: this.id, rectShape: drawingBuffer }, 'layer');
    }
  };

  update = async (arg?: { state: LayerEntity; toolState: CanvasV2State['tool']; isSelected: boolean }) => {
    const state = get(arg, 'state', this.state);
    const toolState = get(arg, 'toolState', this.manager.stateApi.getToolState());
    const isSelected = get(arg, 'isSelected', this.manager.stateApi.getIsSelected(this.id));

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
    await this.updateInteraction({ toolState, isSelected });

    if (this.isFirstRender) {
      await this.updateBbox();
    }

    this.state = state;
    this.isFirstRender = false;
  };

  updateVisibility = (arg?: { isEnabled: boolean }) => {
    this.log.trace('Updating visibility');
    const isEnabled = get(arg, 'isEnabled', this.state.isEnabled);
    const hasObjects = this.objects.size > 0 || this.drawingBuffer !== null;
    this.konva.layer.visible(isEnabled && hasObjects);
  };

  updatePosition = (arg?: { position: Coordinate }) => {
    this.log.trace('Updating position');
    const position = get(arg, 'position', this.state.position);
    const bboxPadding = this.manager.getScaledBboxPadding();

    this.konva.objectGroup.setAttrs({
      x: position.x + this.bbox.x,
      y: position.y + this.bbox.y,
      offsetX: this.bbox.x,
      offsetY: this.bbox.y,
    });
    this.konva.bbox.setAttrs({
      x: position.x + this.bbox.x - bboxPadding,
      y: position.y + this.bbox.y - bboxPadding,
    });
    this.konva.interactionRect.setAttrs({
      x: position.x + this.bbox.x * this.konva.interactionRect.scaleX(),
      y: position.y + this.bbox.y * this.konva.interactionRect.scaleY(),
    });
  };

  updateObjects = async (arg?: { objects: LayerEntity['objects'] }) => {
    this.log.trace('Updating objects');

    const objects = get(arg, 'objects', this.state.objects);

    const objectIds = objects.map(mapId);

    let didUpdate = false;

    // Destroy any objects that are no longer in state
    for (const object of this.objects.values()) {
      if (!objectIds.includes(object.id) && object.id !== this.drawingBuffer?.id) {
        this.objects.delete(object.id);
        object.destroy();
        didUpdate = true;
      }
    }

    for (const obj of objects) {
      if (await this._renderObject(obj)) {
        didUpdate = true;
      }
    }

    if (this.drawingBuffer) {
      if (await this._renderObject(this.drawingBuffer)) {
        didUpdate = true;
      }
    }

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

  updateInteraction = (arg?: { toolState: CanvasV2State['tool']; isSelected: boolean }) => {
    this.log.trace('Updating interaction');

    const toolState = get(arg, 'toolState', this.manager.stateApi.getToolState());
    const isSelected = get(arg, 'isSelected', this.manager.stateApi.getIsSelected(this.id));

    if (this.objects.size === 0) {
      // The layer is totally empty, we can just disable the layer
      this.konva.layer.listening(false);
      return;
    }

    if (isSelected && !this.isTransforming && toolState.selected === 'move') {
      // We are moving this layer, it must be listening
      this.konva.layer.listening(true);

      // The transformer is not needed
      this.transformer.deactivate();

      // The bbox rect should be visible and interaction rect listening for dragging
      this.konva.bbox.visible(true);
      this.konva.interactionRect.listening(true);
    } else if (isSelected && this.isTransforming) {
      // When transforming, we want the stage to still be movable if the view tool is selected. If the transformer or
      // interaction rect are listening, it will interrupt the stage's drag events. So we should disable listening
      // when the view tool is selected
      const listening = toolState.selected !== 'view';
      this.konva.layer.listening(listening);
      this.konva.interactionRect.listening(listening);
      if (listening) {
        this.transformer.activate();
      } else {
        this.transformer.deactivate();
      }

      // Hide the bbox rect, the transformer will has its own bbox
      this.konva.bbox.visible(false);
    } else {
      // The layer is not selected, or we are using a tool that doesn't need the layer to be listening - disable interaction stuff
      this.konva.layer.listening(false);

      // The transformer, bbox and interaction rect should be inactive
      this.transformer.deactivate();
      this.konva.bbox.visible(false);
      this.konva.interactionRect.listening(false);
    }
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
      if (!this.isFirstRender && this.objects.size > 0) {
        // The layer is fully transparent but has objects - reset it
        this.manager.stateApi.onEntityReset({ id: this.id }, 'layer');
      }
      this.konva.bbox.visible(false);
      this.konva.interactionRect.visible(false);
      return;
    }

    this.konva.bbox.visible(true);
    this.konva.interactionRect.visible(true);

    const onePixel = this.manager.getScaledPixel();
    const bboxPadding = this.manager.getScaledBboxPadding();

    this.konva.bbox.setAttrs({
      x: this.state.position.x + this.bbox.x - bboxPadding,
      y: this.state.position.y + this.bbox.y - bboxPadding,
      width: this.bbox.width + bboxPadding * 2,
      height: this.bbox.height + bboxPadding * 2,
      strokeWidth: onePixel,
    });
    this.konva.interactionRect.setAttrs({
      x: this.state.position.x + this.bbox.x,
      y: this.state.position.y + this.bbox.y,
      width: this.bbox.width,
      height: this.bbox.height,
    });
    this.konva.objectGroup.setAttrs({
      x: this.state.position.x + this.bbox.x,
      y: this.state.position.y + this.bbox.y,
      offsetX: this.bbox.x,
      offsetY: this.bbox.y,
    });
  };

  syncStageScale = () => {
    this.log.trace('Syncing scale to stage');

    const onePixel = this.manager.getScaledPixel();
    const bboxPadding = this.manager.getScaledBboxPadding();

    this.konva.bbox.setAttrs({
      x: this.konva.interactionRect.x() - bboxPadding,
      y: this.konva.interactionRect.y() - bboxPadding,
      width: this.konva.interactionRect.width() * this.konva.interactionRect.scaleX() + bboxPadding * 2,
      height: this.konva.interactionRect.height() * this.konva.interactionRect.scaleY() + bboxPadding * 2,
      strokeWidth: onePixel,
    });
  };

  _renderObject = async (obj: LayerEntity['objects'][number], force = false): Promise<boolean> => {
    if (obj.type === 'brush_line') {
      let brushLine = this.objects.get(obj.id);
      assert(brushLine instanceof CanvasBrushLine || brushLine === undefined);

      if (!brushLine) {
        brushLine = new CanvasBrushLine(obj, this);
        this.objects.set(brushLine.id, brushLine);
        this.konva.objectGroup.add(brushLine.konva.group);
        return true;
      } else {
        return await brushLine.update(obj, force);
      }
    } else if (obj.type === 'eraser_line') {
      let eraserLine = this.objects.get(obj.id);
      assert(eraserLine instanceof CanvasEraserLine || eraserLine === undefined);

      if (!eraserLine) {
        eraserLine = new CanvasEraserLine(obj, this);
        this.objects.set(eraserLine.id, eraserLine);
        this.konva.objectGroup.add(eraserLine.konva.group);
        return true;
      } else {
        if (await eraserLine.update(obj, force)) {
          return true;
        }
      }
    } else if (obj.type === 'rect_shape') {
      let rect = this.objects.get(obj.id);
      assert(rect instanceof CanvasRect || rect === undefined);

      if (!rect) {
        rect = new CanvasRect(obj, this);
        this.objects.set(rect.id, rect);
        this.konva.objectGroup.add(rect.konva.group);
        return true;
      } else {
        if (await rect.update(obj, force)) {
          return true;
        }
      }
    } else if (obj.type === 'image') {
      let image = this.objects.get(obj.id);
      assert(image instanceof CanvasImage || image === undefined);

      if (!image) {
        image = new CanvasImage(obj, this);
        this.objects.set(image.id, image);
        this.konva.objectGroup.add(image.konva.group);
        await image.updateImageSource(obj.image.name);
        return true;
      } else {
        if (await image.update(obj, force)) {
          return true;
        }
      }
    }

    return false;
  };

  startTransform = () => {
    this.log.debug('Starting transform');
    this.isTransforming = true;

    // When transforming, we want the stage to still be movable if the view tool is selected. If the transformer or
    // interaction rect are listening, it will interrupt the stage's drag events. So we should disable listening
    // when the view tool is selected
    const listening = this.manager.stateApi.getToolState().selected !== 'view';

    this.konva.layer.listening(listening);
    this.konva.interactionRect.listening(listening);
    this.transformer.activate();

    // Hide the bbox rect, the transformer will has its own bbox
    this.konva.bbox.visible(false);
  };

  resetScale = () => {
    const attrs = {
      scaleX: 1,
      scaleY: 1,
      rotation: 0,
    };
    this.konva.objectGroup.setAttrs(attrs);
    this.konva.bbox.setAttrs(attrs);
    this.konva.interactionRect.setAttrs(attrs);
  };

  rasterizeLayer = async () => {
    this.log.debug('Rasterizing layer');

    const objectGroupClone = this.konva.objectGroup.clone();
    const interactionRectClone = this.konva.interactionRect.clone();
    const rect = interactionRectClone.getClientRect();
    const blob = await konvaNodeToBlob(objectGroupClone, rect);
    if (this.manager._isDebugging) {
      previewBlob(blob, 'Rasterized layer');
    }
    const imageDTO = await uploadImage(blob, `${this.id}_rasterized.png`, 'other', true);
    const { dispatch } = getStore();
    const imageObject = imageDTOToImageObject(imageDTO);
    await this._renderObject(imageObject, true);
    for (const obj of this.objects.values()) {
      if (obj.id !== imageObject.id) {
        obj.konva.group.visible(false);
      }
    }
    this.resetScale();
    dispatch(layerRasterized({ id: this.id, imageObject, position: { x: rect.x, y: rect.y } }));
  };

  stopTransform = () => {
    this.log.debug('Stopping transform');

    this.isTransforming = false;
    this.resetScale();
    this.updatePosition();
    this.updateBbox();
    this.updateInteraction();
  };

  getDefaultRect = (): Rect => {
    return { x: 0, y: 0, width: 0, height: 0 };
  };

  calculateBbox = debounce(() => {
    this.log.debug('Calculating bbox');

    this.isPendingBboxCalculation = true;

    if (this.objects.size === 0) {
      this.log.trace('No objects, resetting bbox');
      this.rect = this.getDefaultRect();
      this.bbox = this.getDefaultRect();
      this.isPendingBboxCalculation = false;
      this.updateBbox();
      return;
    }

    const rect = this.konva.objectGroup.getClientRect({ skipTransform: true });

    /**
     * In some cases, we can use konva's getClientRect as the bbox, but there are some cases where we need to calculate
     * the bbox using pixel data:
     *
     * - Eraser lines are normal lines, except they composite as transparency. Konva's getClientRect includes them when
     *  calculating the bbox.
     * - Clipped portions of lines will be included in the client rect.
     * - Images have transparency, so they will be included in the client rect.
     *
     * TODO(psyche): Using pixel data is slow. Is it possible to be clever and somehow subtract the eraser lines and
     * clipped areas from the client rect?
     */
    let needsPixelBbox = false;
    for (const obj of this.objects.values()) {
      const isEraserLine = obj instanceof CanvasEraserLine;
      const isImage = obj instanceof CanvasImage;
      const hasClip = obj instanceof CanvasBrushLine && obj.state.clip;
      if (isEraserLine || hasClip || isImage) {
        needsPixelBbox = true;
        break;
      }
    }

    if (!needsPixelBbox) {
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
          this.bbox = this.getDefaultRect();
          this.rect = this.getDefaultRect();
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
      isFirstRender: this.isFirstRender,
      isTransforming: this.isTransforming,
      isPendingBboxCalculation: this.isPendingBboxCalculation,
      objects: Array.from(this.objects.values()).map((obj) => obj.repr()),
    };
  };

  logDebugInfo(msg = 'Debug info') {
    const info = {
      repr: this.repr(),
      interactionRectAttrs: {
        x: this.konva.interactionRect.x(),
        y: this.konva.interactionRect.y(),
        scaleX: this.konva.interactionRect.scaleX(),
        scaleY: this.konva.interactionRect.scaleY(),
        width: this.konva.interactionRect.width(),
        height: this.konva.interactionRect.height(),
        rotation: this.konva.interactionRect.rotation(),
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
