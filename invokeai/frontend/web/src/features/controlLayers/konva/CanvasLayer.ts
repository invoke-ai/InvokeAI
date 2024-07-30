import { getStore } from 'app/store/nanostores/store';
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

  private drawingBuffer: BrushLine | EraserLine | RectShape | null;
  private state: LayerEntity;

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

  offsetX: number;
  offsetY: number;
  width: number;
  height: number;
  log: Logger;
  bboxNeedsUpdate: boolean;
  isTransforming: boolean;
  isFirstRender: boolean;

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
      console.log('>>> transformstart');
      console.log('interactionRect', {
        x: this.konva.interactionRect.x(),
        y: this.konva.interactionRect.y(),
        scaleX: this.konva.interactionRect.scaleX(),
        scaleY: this.konva.interactionRect.scaleY(),
        width: this.konva.interactionRect.width(),
        height: this.konva.interactionRect.height(),
      });
      this.logBbox('transformstart bbox');
      console.log('this.state.position', this.state.position);
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

      console.log('>>> transform');
      console.log('activeAnchor', this.konva.transformer.getActiveAnchor());
      console.log('interactionRect', {
        x: this.konva.interactionRect.x(),
        y: this.konva.interactionRect.y(),
        scaleX: this.konva.interactionRect.scaleX(),
        scaleY: this.konva.interactionRect.scaleY(),
        width: this.konva.interactionRect.width(),
        height: this.konva.interactionRect.height(),
        rotation: this.konva.interactionRect.rotation(),
      });

      this.konva.objectGroup.setAttrs({
        x: this.konva.interactionRect.x() - this.offsetX * this.konva.interactionRect.scaleX(),
        y: this.konva.interactionRect.y() - this.offsetY * this.konva.interactionRect.scaleY(),
        scaleX: this.konva.interactionRect.scaleX(),
        scaleY: this.konva.interactionRect.scaleY(),
        rotation: this.konva.interactionRect.rotation(),
      });

      console.log('objectGroup', {
        x: this.konva.objectGroup.x(),
        y: this.konva.objectGroup.y(),
        scaleX: this.konva.objectGroup.scaleX(),
        scaleY: this.konva.objectGroup.scaleY(),
        offsetX: this.offsetX,
        offsetY: this.offsetY,
        width: this.konva.objectGroup.width(),
        height: this.konva.objectGroup.height(),
        rotation: this.konva.objectGroup.rotation(),
      });
    });

    this.konva.transformer.on('transformend', () => {
      // this.offsetX = this.konva.interactionRect.x() - this.state.position.x;
      // this.offsetY = this.konva.interactionRect.y() - this.state.position.y;
      // this.width = Math.round(this.konva.interactionRect.width() * this.konva.interactionRect.scaleX());
      // this.height = Math.round(this.konva.interactionRect.height() * this.konva.interactionRect.scaleY());
      // this.manager.stateApi.onPosChanged(
      //   {
      //     id: this.id,
      //     position: { x: this.konva.objectGroup.x(), y: this.konva.objectGroup.y() },
      //   },
      //   'layer'
      // );
      this.logBbox('transformend bbox');
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
        x: this.konva.interactionRect.x() - this.offsetX * this.konva.interactionRect.scaleX(),
        y: this.konva.interactionRect.y() - this.offsetY * this.konva.interactionRect.scaleY(),
      });
    });
    this.konva.interactionRect.on('dragend', () => {
      this.logBbox('dragend bbox');

      if (this.isTransforming) {
        // When the user cancels the transformation, we need to reset the layer, so we should not update the layer's
        // positition while we are transforming - bail out early.
        return;
      }

      this.manager.stateApi.onPosChanged(
        {
          id: this.id,
          position: {
            x: this.konva.interactionRect.x() - this.offsetX * this.konva.interactionRect.scaleX(),
            y: this.konva.interactionRect.y() - this.offsetY * this.konva.interactionRect.scaleY(),
          },
        },
        'layer'
      );
    });

    this.objects = new Map();
    this.drawingBuffer = null;
    this.state = state;
    this.offsetX = 0;
    this.offsetY = 0;
    this.width = 0;
    this.height = 0;
    this.bboxNeedsUpdate = true;
    this.isTransforming = false;
    this.isFirstRender = true;
    this.log = this.manager.getLogger(`layer_${this.id}`);

    console.log(this);
  }

  destroy(): void {
    this.log.debug(`Layer ${this.id} - destroying`);
    this.konva.layer.destroy();
  }

  getDrawingBuffer() {
    return this.drawingBuffer;
  }
  async setDrawingBuffer(obj: BrushLine | EraserLine | RectShape | null) {
    if (obj) {
      this.drawingBuffer = obj;
      await this._renderObject(this.drawingBuffer, true);
    } else {
      this.drawingBuffer = null;
    }
  }

  async finalizeDrawingBuffer() {
    if (!this.drawingBuffer) {
      return;
    }
    const drawingBuffer = this.drawingBuffer;
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
    const state = get(arg, 'state', this.state);
    const toolState = get(arg, 'toolState', this.manager.stateApi.getToolState());
    const isSelected = get(arg, 'isSelected', this.manager.stateApi.getIsSelected(this.id));

    if (!this.isFirstRender && state === this.state) {
      this.log.trace('State unchanged, skipping update');
      return;
    }

    this.log.debug('Updating');
    const { position, objects, opacity, isEnabled } = state;

    if (this.isFirstRender || position !== this.state.position) {
      await this.updatePosition({ position });
    }
    if (this.isFirstRender || objects !== this.state.objects) {
      await this.updateObjects({ objects });
    }
    if (this.isFirstRender || opacity !== this.state.opacity) {
      await this.updateOpacity({ opacity });
    }
    if (this.isFirstRender || isEnabled !== this.state.isEnabled) {
      await this.updateVisibility({ isEnabled });
    }
    await this.updateInteraction({ toolState, isSelected });
    this.state = state;
  }

  async updateVisibility(arg?: { isEnabled: boolean }) {
    this.log.trace('Updating visibility');
    const isEnabled = get(arg, 'isEnabled', this.state.isEnabled);
    const hasObjects = this.objects.size > 0 || this.drawingBuffer !== null;
    this.konva.layer.visible(isEnabled || hasObjects);
  }

  async updatePosition(arg?: { position: Coordinate }) {
    this.log.trace('Updating position');
    const position = get(arg, 'position', this.state.position);
    const bboxPadding = this.manager.getScaledBboxPadding();

    this.konva.objectGroup.setAttrs({
      x: position.x,
      y: position.y,
    });
    this.konva.bbox.setAttrs({
      x: position.x + this.offsetX * this.konva.interactionRect.scaleX() - bboxPadding,
      y: position.y + this.offsetY * this.konva.interactionRect.scaleY() - bboxPadding,
    });
    this.konva.interactionRect.setAttrs({
      x: position.x + this.offsetX * this.konva.interactionRect.scaleX(),
      y: position.y + this.offsetY * this.konva.interactionRect.scaleY(),
    });
  }

  async updateObjects(arg?: { objects: LayerEntity['objects'] }) {
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
  }

  async updateOpacity(arg?: { opacity: number }) {
    this.log.trace('Updating opacity');

    const opacity = get(arg, 'opacity', this.state.opacity);

    this.konva.objectGroup.opacity(opacity);
  }

  async updateInteraction(arg?: { toolState: CanvasV2State['tool']; isSelected: boolean }) {
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
    this.log.trace('Updating bbox');

    // If the bbox has no width or height, that means the layer is fully transparent. This can happen if it is only
    // eraser lines, fully clipped brush lines or if it has been fully erased. In this case, we should reset the layer
    // so we aren't drawing shapes that do not render anything.
    if (this.width === 0 || this.height === 0) {
      this.manager.stateApi.onEntityReset({ id: this.id }, 'layer');
      return;
    }

    const onePixel = this.manager.getScaledPixel();
    const bboxPadding = this.manager.getScaledBboxPadding();

    this.konva.bbox.setAttrs({
      x: this.state.position.x + this.offsetX * this.konva.interactionRect.scaleX() - bboxPadding,
      y: this.state.position.y + this.offsetY * this.konva.interactionRect.scaleY() - bboxPadding,
      width: this.width + bboxPadding * 2,
      height: this.height + bboxPadding * 2,
      strokeWidth: onePixel,
    });
    this.konva.interactionRect.setAttrs({
      x: this.state.position.x + this.offsetX * this.konva.interactionRect.scaleX(),
      y: this.state.position.y + this.offsetY * this.konva.interactionRect.scaleY(),
      width: this.width,
      height: this.height,
    });
  }

  async syncStageScale() {
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
    this.log.debug('Starting transform');
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
    this.konva.objectGroup.scaleX(1);
    this.konva.objectGroup.scaleY(1);
    this.konva.bbox.scaleX(1);
    this.konva.bbox.scaleY(1);
    this.konva.interactionRect.scaleX(1);
    this.konva.interactionRect.scaleY(1);
  }

  async applyTransform() {
    this.log.debug('Applying transform');

    this.isTransforming = false;
    const objectGroupClone = this.konva.objectGroup.clone();
    const rect = {
      x: this.konva.interactionRect.x(),
      y: this.konva.interactionRect.y(),
      width: this.konva.interactionRect.width() * this.konva.interactionRect.scaleX(),
      height: this.konva.interactionRect.height() * this.konva.interactionRect.scaleY(),
    };
    const blob = await konvaNodeToBlob(objectGroupClone, rect);
    previewBlob(blob, 'transformed layer');
    const imageDTO = await uploadImage(blob, `${this.id}_transform.png`, 'other', true, true);
    const { dispatch } = getStore();
    dispatch(layerRasterized({ id: this.id, imageDTO, position: this.konva.interactionRect.position() }));
    this.isTransforming = false;
    this.resetScale();
  }

  async cancelTransform() {
    this.log.debug('Canceling transform');

    this.isTransforming = false;
    this.resetScale();
    await this.updatePosition({ position: this.state.position });
    await this.updateBbox();
    await this.updateInteraction({
      toolState: this.manager.stateApi.getToolState(),
      isSelected: this.manager.stateApi.getIsSelected(this.id),
    });
  }

  calculateBbox = debounce(() => {
    this.log.debug('Calculating bbox');

    if (this.objects.size === 0) {
      this.offsetX = 0;
      this.offsetY = 0;
      this.width = 0;
      this.height = 0;
      this.updateBbox();
      return;
    }

    let needsPixelBbox = false;
    const rect = this.konva.objectGroup.getClientRect({ skipTransform: true });

    console.log('getBbox rect', rect);

    /**
     * In some cases, we can use konva's getClientRect as the bbox, but there are some cases where we need to calculate
     * the bbox using pixel data:
     *
     * - Eraser lines are normal lines, except they composite as transparency. Konva's getClientRect includes them when
     *  calculating the bbox.
     * - Clipped portions of lines will be included in the client rect.
     *
     * TODO(psyche): Using pixel data is slow. Is it possible to be clever and somehow subtract the eraser lines and
     * clipped areas from the client rect?
     */
    for (const obj of this.objects.values()) {
      const isEraserLine = obj instanceof CanvasEraserLine;
      const hasClip = obj instanceof CanvasBrushLine && obj.state.clip;
      if (isEraserLine || hasClip) {
        needsPixelBbox = true;
        break;
      }
    }

    if (!needsPixelBbox) {
      this.offsetX = rect.x;
      this.offsetY = rect.y;
      this.width = rect.width;
      this.height = rect.height;
      this.logBbox('new bbox from client rect');
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
        console.log('extents', extents);
        if (extents) {
          const { minX, minY, maxX, maxY } = extents;
          this.offsetX = minX + rect.x;
          this.offsetY = minY + rect.y;
          this.width = maxX - minX;
          this.height = maxY - minY;
        } else {
          this.offsetX = 0;
          this.offsetY = 0;
          this.width = 0;
          this.height = 0;
        }
        this.logBbox('new bbox from worker');
        this.updateBbox();
        clone.destroy();
      }
    );
  }, CanvasManager.BBOX_DEBOUNCE_MS);

  logBbox(msg: string = 'bbox') {
    console.log(msg, {
      x: this.state.position.x,
      y: this.state.position.y,
      offsetX: this.offsetX,
      offsetY: this.offsetY,
      width: this.width,
      height: this.height,
    });
  }

  getLayerRect() {
    return {
      x: this.state.position.x + this.offsetX,
      y: this.state.position.y + this.offsetY,
      width: this.width,
      height: this.height,
    };
  }
}
