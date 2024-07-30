import { getStore } from 'app/store/nanostores/store';
import { deepClone } from 'common/util/deepClone';
import { CanvasBrushLine } from 'features/controlLayers/konva/CanvasBrushLine';
import { CanvasEraserLine } from 'features/controlLayers/konva/CanvasEraserLine';
import { CanvasImage } from 'features/controlLayers/konva/CanvasImage';
import { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasRect } from 'features/controlLayers/konva/CanvasRect';
import { getBrushLineId, getEraserLineId, getRectShapeId } from 'features/controlLayers/konva/naming';
import { konvaNodeToBlob, mapId, previewBlob } from 'features/controlLayers/konva/util';
import { layerRasterized } from 'features/controlLayers/store/canvasV2Slice';
import type {
  BrushLine,
  CanvasV2State,
  Coordinate,
  EraserLine,
  LayerEntity,
  Rect,
  RectShape,
} from 'features/controlLayers/store/types';
import Konva from 'konva';
import { debounce, get } from 'lodash-es';
import type { Logger } from 'roarr';
import { uploadImage } from 'services/api/endpoints/images';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

export class CanvasLayer {
  static NAME_PREFIX = 'layer';
  static LAYER_NAME = `${CanvasLayer.NAME_PREFIX}_layer`;
  static TRANSFORMER_NAME = `${CanvasLayer.NAME_PREFIX}_transformer`;
  static INTERACTION_RECT_NAME = `${CanvasLayer.NAME_PREFIX}_interaction-rect`;
  static GROUP_NAME = `${CanvasLayer.NAME_PREFIX}_group`;
  static OBJECT_GROUP_NAME = `${CanvasLayer.NAME_PREFIX}_object-group`;
  static BBOX_NAME = `${CanvasLayer.NAME_PREFIX}_bbox`;

  _drawingBuffer: BrushLine | EraserLine | RectShape | null;
  _state: LayerEntity;

  id: string;
  manager: CanvasManager;

  konva: {
    layer: Konva.Layer;
    bbox: Konva.Rect;
    objectGroup: Konva.Group;
    transformer: Konva.Transformer;
    interactionRect: Konva.Rect;
  };
  objects: Map<string, CanvasBrushLine | CanvasEraserLine | CanvasRect | CanvasImage>;

  _log: Logger;
  _bboxNeedsUpdate: boolean;
  _isFirstRender: boolean;

  isTransforming: boolean;

  rect: Rect;
  bbox: Rect;

  constructor(state: LayerEntity, manager: CanvasManager) {
    this.id = state.id;
    this.manager = manager;
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
      transformer: new Konva.Transformer({
        name: CanvasLayer.TRANSFORMER_NAME,
        draggable: false,
        // enabledAnchors: ['top-left', 'top-right', 'bottom-left', 'bottom-right'],
        rotateEnabled: true,
        flipEnabled: true,
        listening: false,
        padding: this.manager.getTransformerPadding(),
        stroke: 'hsl(200deg 76% 59%)', // invokeBlue.400
        keepRatio: false,
      }),
      interactionRect: new Konva.Rect({
        name: CanvasLayer.INTERACTION_RECT_NAME,
        listening: false,
        draggable: true,
        // fill: 'rgba(255,0,0,0.5)',
      }),
    };

    this.konva.layer.add(this.konva.objectGroup);
    this.konva.layer.add(this.konva.transformer);
    this.konva.layer.add(this.konva.interactionRect);
    this.konva.layer.add(this.konva.bbox);

    this.konva.transformer.on('transformstart', () => {
      this.logDebugInfo("'transformstart' fired");
    });

    this.konva.transformer.on('transform', () => {
      // Always snap the interaction rect to the nearest pixel when transforming

      // const x = Math.round(this.konva.interactionRect.x());
      // const y = Math.round(this.konva.interactionRect.y());
      // // Snap its position
      // this.konva.interactionRect.x(x);
      // this.konva.interactionRect.y(y);

      // // Calculate the new scale of the interaction rect such that its width and height snap to the nearest pixel
      // const targetWidth = Math.max(
      //   Math.round(this.konva.interactionRect.width() * Math.abs(this.konva.interactionRect.scaleX())),
      //   MIN_LAYER_SIZE_PX
      // );
      // const scaleX = targetWidth / this.konva.interactionRect.width();
      // const targetHeight = Math.max(
      //   Math.round(this.konva.interactionRect.height() * Math.abs(this.konva.interactionRect.scaleY())),
      //   MIN_LAYER_SIZE_PX
      // );
      // const scaleY = targetHeight / this.konva.interactionRect.height();

      // // Snap the width and height (via scale) of the interaction rect
      // this.konva.interactionRect.scaleX(scaleX);
      // this.konva.interactionRect.scaleY(scaleY);
      // this.konva.interactionRect.rotation(0);

      this.logDebugInfo("'transform' fired");

      this.konva.objectGroup.setAttrs({
        x: this.konva.interactionRect.x(),
        y: this.konva.interactionRect.y(),
        scaleX: this.konva.interactionRect.scaleX(),
        scaleY: this.konva.interactionRect.scaleY(),
        rotation: this.konva.interactionRect.rotation(),
      });
    });

    this.konva.transformer.on('transformend', () => {
      this.logDebugInfo("'transformend' fired");
    });

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
      this.logDebugInfo("'dragend' fired");

      if (this.isTransforming) {
        // When the user cancels the transformation, we need to reset the layer, so we should not update the layer's
        // positition while we are transforming - bail out early.
        return;
      }

      this.manager.stateApi.onPosChanged(
        {
          id: this.id,
          position: {
            x: this.konva.interactionRect.x() - this.bbox.x,
            y: this.konva.interactionRect.y() - this.bbox.y,
          },
        },
        'layer'
      );
    });

    this.objects = new Map();
    this._drawingBuffer = null;
    this._state = state;
    this.rect = this.getDefaultRect();
    this.bbox = this.getDefaultRect();
    this._bboxNeedsUpdate = true;
    this.isTransforming = false;
    this._isFirstRender = true;
    this._log = this.manager.getLogger(`layer_${this.id}`);
  }

  destroy(): void {
    this._log.debug(`Layer ${this.id} - destroying`);
    this.konva.layer.destroy();
  }

  getDrawingBuffer() {
    return this._drawingBuffer;
  }
  async setDrawingBuffer(obj: BrushLine | EraserLine | RectShape | null) {
    if (obj) {
      this._drawingBuffer = obj;
      await this._renderObject(this._drawingBuffer, true);
    } else {
      this._drawingBuffer = null;
    }
  }

  async finalizeDrawingBuffer() {
    if (!this._drawingBuffer) {
      return;
    }
    const drawingBuffer = this._drawingBuffer;
    this.setDrawingBuffer(null);

    // We need to give the objects a fresh ID else they will be considered the same object when they are re-rendered as
    // a non-buffer object, and we won't trigger things like bbox calculation

    if (drawingBuffer.type === 'brush_line') {
      drawingBuffer.id = getBrushLineId(this.id, uuidv4());
      this.manager.stateApi.onBrushLineAdded({ id: this.id, brushLine: drawingBuffer }, 'layer');
    } else if (drawingBuffer.type === 'eraser_line') {
      drawingBuffer.id = getEraserLineId(this.id, uuidv4());
      this.manager.stateApi.onEraserLineAdded({ id: this.id, eraserLine: drawingBuffer }, 'layer');
    } else if (drawingBuffer.type === 'rect_shape') {
      drawingBuffer.id = getRectShapeId(this.id, uuidv4());
      this.manager.stateApi.onRectShapeAdded({ id: this.id, rectShape: drawingBuffer }, 'layer');
    }
  }

  async update(arg?: { state: LayerEntity; toolState: CanvasV2State['tool']; isSelected: boolean }) {
    const state = get(arg, 'state', this._state);
    const toolState = get(arg, 'toolState', this.manager.stateApi.getToolState());
    const isSelected = get(arg, 'isSelected', this.manager.stateApi.getIsSelected(this.id));

    if (!this._isFirstRender && state === this._state) {
      this._log.trace('State unchanged, skipping update');
      return;
    }

    this._log.debug('Updating');
    const { position, objects, opacity, isEnabled } = state;

    if (this._isFirstRender || position !== this._state.position) {
      await this.updatePosition({ position });
    }
    if (this._isFirstRender || objects !== this._state.objects) {
      await this.updateObjects({ objects });
    }
    if (this._isFirstRender || opacity !== this._state.opacity) {
      await this.updateOpacity({ opacity });
    }
    if (this._isFirstRender || isEnabled !== this._state.isEnabled) {
      await this.updateVisibility({ isEnabled });
    }
    await this.updateInteraction({ toolState, isSelected });

    if (this._isFirstRender) {
      await this.updateBbox();
    }

    this._state = state;
    this._isFirstRender = false;
  }

  async updateVisibility(arg?: { isEnabled: boolean }) {
    this._log.trace('Updating visibility');
    const isEnabled = get(arg, 'isEnabled', this._state.isEnabled);
    const hasObjects = this.objects.size > 0 || this._drawingBuffer !== null;
    this.konva.layer.visible(isEnabled || hasObjects);
  }

  async updatePosition(arg?: { position: Coordinate }) {
    this._log.trace('Updating position');
    const position = get(arg, 'position', this._state.position);
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
  }

  async updateObjects(arg?: { objects: LayerEntity['objects'] }) {
    this._log.trace('Updating objects');

    const objects = get(arg, 'objects', this._state.objects);

    const objectIds = objects.map(mapId);

    let didUpdate = false;

    // Destroy any objects that are no longer in state
    for (const object of this.objects.values()) {
      if (!objectIds.includes(object.id) && object.id !== this._drawingBuffer?.id) {
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

    if (this._drawingBuffer) {
      if (await this._renderObject(this._drawingBuffer)) {
        didUpdate = true;
      }
    }

    if (didUpdate) {
      this.calculateBbox();
    }
  }

  async updateOpacity(arg?: { opacity: number }) {
    this._log.trace('Updating opacity');

    const opacity = get(arg, 'opacity', this._state.opacity);

    this.konva.objectGroup.opacity(opacity);
  }

  async updateInteraction(arg?: { toolState: CanvasV2State['tool']; isSelected: boolean }) {
    this._log.trace('Updating interaction');

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
      this.konva.transformer.listening(false);
      this.konva.transformer.nodes([]);

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
      this.konva.transformer.listening(listening);

      // The transformer transforms the interaction rect, not the object group
      this.konva.transformer.nodes([this.konva.interactionRect]);

      // Hide the bbox rect, the transformer will has its own bbox
      this.konva.bbox.visible(false);
    } else {
      // The layer is not selected, or we are using a tool that doesn't need the layer to be listening - disable interaction stuff
      this.konva.layer.listening(false);

      // The transformer, bbox and interaction rect should be inactive
      this.konva.transformer.listening(false);
      this.konva.transformer.nodes([]);
      this.konva.bbox.visible(false);
      this.konva.interactionRect.listening(false);
    }
  }

  async updateBbox() {
    this._log.trace('Updating bbox');

    // If the bbox has no width or height, that means the layer is fully transparent. This can happen if it is only
    // eraser lines, fully clipped brush lines or if it has been fully erased.
    if (this.bbox.width === 0 || this.bbox.height === 0) {
      if (this.objects.size > 0) {
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
      x: this._state.position.x + this.bbox.x - bboxPadding,
      y: this._state.position.y + this.bbox.y - bboxPadding,
      width: this.bbox.width + bboxPadding * 2,
      height: this.bbox.height + bboxPadding * 2,
      strokeWidth: onePixel,
    });
    this.konva.interactionRect.setAttrs({
      x: this._state.position.x + this.bbox.x,
      y: this._state.position.y + this.bbox.y,
      width: this.bbox.width,
      height: this.bbox.height,
    });
    this.konva.objectGroup.setAttrs({
      x: this._state.position.x + this.bbox.x,
      y: this._state.position.y + this.bbox.y,
      offsetX: this.bbox.x,
      offsetY: this.bbox.y,
    });
  }

  async syncStageScale() {
    this._log.trace('Syncing scale to stage');

    const onePixel = this.manager.getScaledPixel();
    const bboxPadding = this.manager.getScaledBboxPadding();

    this.konva.bbox.setAttrs({
      x: this.konva.interactionRect.x() - bboxPadding,
      y: this.konva.interactionRect.y() - bboxPadding,
      width: this.konva.interactionRect.width() * this.konva.interactionRect.scaleX() + bboxPadding * 2,
      height: this.konva.interactionRect.height() * this.konva.interactionRect.scaleY() + bboxPadding * 2,
      strokeWidth: onePixel,
    });
    this.konva.transformer.forceUpdate();
  }

  async _renderObject(obj: LayerEntity['objects'][number], force = false): Promise<boolean> {
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
  }

  async startTransform() {
    this._log.debug('Starting transform');
    this.isTransforming = true;

    // When transforming, we want the stage to still be movable if the view tool is selected. If the transformer or
    // interaction rect are listening, it will interrupt the stage's drag events. So we should disable listening
    // when the view tool is selected
    const listening = this.manager.stateApi.getToolState().selected !== 'view';

    this.konva.layer.listening(listening);
    this.konva.interactionRect.listening(listening);
    this.konva.transformer.listening(listening);

    // The transformer transforms the interaction rect, not the object group
    this.konva.transformer.nodes([this.konva.interactionRect]);

    // Hide the bbox rect, the transformer will has its own bbox
    this.konva.bbox.visible(false);
  }

  async resetScale() {
    const attrs = {
      scaleX: 1,
      scaleY: 1,
      rotation: 0,
    };
    this.konva.objectGroup.setAttrs(attrs);
    this.konva.bbox.setAttrs(attrs);
    this.konva.interactionRect.setAttrs(attrs);
  }

  async applyTransform() {
    this._log.debug('Applying transform');

    const objectGroupClone = this.konva.objectGroup.clone();
    const interactionRectClone = this.konva.interactionRect.clone();
    const rect = interactionRectClone.getClientRect();
    const blob = await konvaNodeToBlob(objectGroupClone, rect);
    if (this.manager._isDebugging) {
      previewBlob(blob, 'transformed layer');
    }
    const imageDTO = await uploadImage(blob, `${this.id}_transform.png`, 'other', true);
    const { dispatch } = getStore();
    dispatch(layerRasterized({ id: this.id, imageDTO, position: { x: rect.x, y: rect.y } }));
    this.isTransforming = false;
    this.resetScale();
  }

  async cancelTransform() {
    this._log.debug('Canceling transform');

    this.isTransforming = false;
    this.resetScale();
    await this.updatePosition({ position: this._state.position });
    await this.updateBbox();
    await this.updateInteraction({
      toolState: this.manager.stateApi.getToolState(),
      isSelected: this.manager.stateApi.getIsSelected(this.id),
    });
  }

  getDefaultRect(): Rect {
    return { x: 0, y: 0, width: 0, height: 0 };
  }

  calculateBbox = debounce(() => {
    this._log.debug('Calculating bbox');

    if (this.objects.size === 0) {
      this.rect = this.getDefaultRect();
      this.bbox = this.getDefaultRect();
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
      this._log.trace({ bbox: this.bbox, rect: this.rect }, 'Got bbox from client rect');
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
        this.rect = deepClone(rect);
        if (extents) {
          const { minX, minY, maxX, maxY } = extents;
          this.bbox = {
            x: rect.x + minX,
            y: rect.y + minY,
            width: maxX - minX,
            height: maxY - minY,
          };
        } else {
          this.bbox = deepClone(rect);
        }
        this._log.trace({ bbox: this.bbox, rect: this.rect, extents }, `Got bbox from worker`);
        this.updateBbox();
        clone.destroy();
      }
    );
  }, CanvasManager.BBOX_DEBOUNCE_MS);

  logDebugInfo(msg = 'Debug info') {
    const debugInfo = {
      id: this.id,
      state: this._state,
      rect: this.rect,
      bbox: this.bbox,
      objects: Array.from(this.objects.values()).map((obj) => obj.id),
      isTransforming: this.isTransforming,
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
    this._log.debug(debugInfo, msg);
  }
}
