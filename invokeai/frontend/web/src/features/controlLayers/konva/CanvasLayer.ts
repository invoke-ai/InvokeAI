import { CanvasBrushLine } from 'features/controlLayers/konva/CanvasBrushLine';
import { CanvasEraserLine } from 'features/controlLayers/konva/CanvasEraserLine';
import { CanvasImage } from 'features/controlLayers/konva/CanvasImage';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasRect } from 'features/controlLayers/konva/CanvasRect';
import { mapId } from 'features/controlLayers/konva/util';
import type { BrushLine, EraserLine, LayerEntity, Rect, RectShape } from 'features/controlLayers/store/types';
import { isDrawingTool } from 'features/controlLayers/store/types';
import Konva from 'konva';
import { debounce } from 'lodash-es';
import { assert } from 'tsafe';

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
    group: Konva.Group;
    bbox: Konva.Rect;

    objectGroup: Konva.Group;
    transformer: Konva.Transformer;
    interactionRect: Konva.Rect;
  };
  objects: Map<string, CanvasBrushLine | CanvasEraserLine | CanvasRect | CanvasImage>;
  bbox: Rect;

  getBbox = debounce(this._getBbox, 300);

  constructor(state: LayerEntity, manager: CanvasManager) {
    this.id = state.id;
    this.manager = manager;
    this.konva = {
      layer: new Konva.Layer({ id: this.id, name: CanvasLayer.LAYER_NAME, listening: false }),
      group: new Konva.Group({ name: CanvasLayer.GROUP_NAME, listening: true, draggable: true }),
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
        enabledAnchors: ['top-left', 'top-right', 'bottom-left', 'bottom-right'],
        rotateEnabled: false,
        flipEnabled: false,
        listening: false,
      }),
      interactionRect: new Konva.Rect({
        name: CanvasLayer.INTERACTION_RECT_NAME,
        listening: false,
        draggable: false,
        fill: 'rgba(255,0,0,0.5)',
      }),
    };

    this.konva.layer.add(this.konva.group);
    this.konva.layer.add(this.konva.transformer);
    this.konva.group.add(this.konva.objectGroup);
    this.konva.group.add(this.konva.interactionRect);
    this.konva.group.add(this.konva.bbox);

    // this.konva.transformer.on('transform', () => {
    //   console.log(this.konva.interactionRect.position());
    //   this.konva.objectGroup.setAttrs({
    //     scaleX: this.konva.interactionRect.scaleX(),
    //     scaleY: this.konva.interactionRect.scaleY(),
    //     // rotation: this.konva.interactionRect.rotation(),
    //     x: this.konva.interactionRect.x(),
    //     t: this.konva.interactionRect.y(),
    //   });
    // });
    this.konva.transformer.on('transformend', () => {
      console.log(this.bbox);
      this.bbox = {
        x: this.bbox.x * this.konva.group.scaleX(),
        y: this.bbox.y * this.konva.group.scaleY(),
        width: this.bbox.width * this.konva.group.scaleX(),
        height: this.bbox.height * this.konva.group.scaleY(),
      };
      console.log(this.bbox);
      this.renderBbox();
      this.manager.stateApi.onScaleChanged(
        {
          id: this.id,
          scale: this.konva.group.scaleX(),
          position: { x: this.konva.group.x(), y: this.konva.group.y() },
        },
        'layer'
      );
    });
    this.konva.group.on('dragend', () => {
      this.manager.stateApi.onPosChanged(
        { id: this.id, position: { x: this.konva.group.x(), y: this.konva.group.y() } },
        'layer'
      );
    });

    this.objects = new Map();
    this.drawingBuffer = null;
    this.state = state;
    this.bbox = CanvasLayer.DEFAULT_BBOX_RECT;
  }

  private static get DEFAULT_BBOX_RECT() {
    return { x: 0, y: 0, width: 0, height: 0 };
  }

  destroy(): void {
    this.konva.layer.destroy();
  }

  getDrawingBuffer() {
    return this.drawingBuffer;
  }

  async setDrawingBuffer(obj: BrushLine | EraserLine | RectShape | null) {
    if (obj) {
      this.drawingBuffer = obj;
      await this.renderObject(this.drawingBuffer, true);
      this.updateGroup(true);
    } else {
      this.drawingBuffer = null;
    }
  }

  finalizeDrawingBuffer() {
    if (!this.drawingBuffer) {
      return;
    }
    if (this.drawingBuffer.type === 'brush_line') {
      this.manager.stateApi.onBrushLineAdded({ id: this.id, brushLine: this.drawingBuffer }, 'layer');
    } else if (this.drawingBuffer.type === 'eraser_line') {
      this.manager.stateApi.onEraserLineAdded({ id: this.id, eraserLine: this.drawingBuffer }, 'layer');
    } else if (this.drawingBuffer.type === 'rect_shape') {
      this.manager.stateApi.onRectShapeAdded({ id: this.id, rectShape: this.drawingBuffer }, 'layer');
    }
    this.setDrawingBuffer(null);
  }

  async render(state: LayerEntity) {
    this.state = state;

    // Update the layer's position and listening state
    this.konva.group.setAttrs({
      x: state.position.x,
      y: state.position.y,
      scaleX: 1,
      scaleY: 1,
    });

    let didDraw = false;

    const objectIds = state.objects.map(mapId);
    // Destroy any objects that are no longer in state
    for (const object of this.objects.values()) {
      if (!objectIds.includes(object.id) && object.id !== this.drawingBuffer?.id) {
        this.objects.delete(object.id);
        object.destroy();
        didDraw = true;
      }
    }

    for (const obj of state.objects) {
      if (await this.renderObject(obj)) {
        didDraw = true;
      }
    }

    if (this.drawingBuffer) {
      if (await this.renderObject(this.drawingBuffer)) {
        didDraw = true;
      }
    }

    this.renderBbox();
    this.updateGroup(didDraw);
  }

  private async renderObject(obj: LayerEntity['objects'][number], force = false): Promise<boolean> {
    if (obj.type === 'brush_line') {
      let brushLine = this.objects.get(obj.id);
      assert(brushLine instanceof CanvasBrushLine || brushLine === undefined);

      if (!brushLine) {
        brushLine = new CanvasBrushLine(obj);
        this.objects.set(brushLine.id, brushLine);
        this.konva.objectGroup.add(brushLine.konva.group);
        return true;
      } else {
        if (brushLine.update(obj, force)) {
          return true;
        }
      }
    } else if (obj.type === 'eraser_line') {
      let eraserLine = this.objects.get(obj.id);
      assert(eraserLine instanceof CanvasEraserLine || eraserLine === undefined);

      if (!eraserLine) {
        eraserLine = new CanvasEraserLine(obj);
        this.objects.set(eraserLine.id, eraserLine);
        this.konva.objectGroup.add(eraserLine.konva.group);
        return true;
      } else {
        if (eraserLine.update(obj, force)) {
          return true;
        }
      }
    } else if (obj.type === 'rect_shape') {
      let rect = this.objects.get(obj.id);
      assert(rect instanceof CanvasRect || rect === undefined);

      if (!rect) {
        rect = new CanvasRect(obj);
        this.objects.set(rect.id, rect);
        this.konva.objectGroup.add(rect.konva.group);
        return true;
      } else {
        if (rect.update(obj, force)) {
          return true;
        }
      }
    } else if (obj.type === 'image') {
      let image = this.objects.get(obj.id);
      assert(image instanceof CanvasImage || image === undefined);

      if (!image) {
        image = new CanvasImage(obj);
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

  updateGroup(didDraw: boolean) {
    if (!this.state.isEnabled) {
      this.konva.layer.visible(false);
      return;
    }

    if (didDraw) {
      if (this.objects.size > 0) {
        this.getBbox();
      } else {
        this.bbox = CanvasLayer.DEFAULT_BBOX_RECT;
        this.renderBbox();
      }
    }

    this.konva.layer.visible(true);
    this.konva.objectGroup.opacity(this.state.opacity);
    const isSelected = this.manager.stateApi.getIsSelected(this.id);
    const selectedTool = this.manager.stateApi.getToolState().selected;

    const isTransforming = selectedTool === 'transform' && isSelected;
    const isMoving = selectedTool === 'move' && isSelected;

    this.konva.layer.listening(isTransforming || isMoving);
    this.konva.transformer.listening(isTransforming);
    this.konva.bbox.visible(isMoving);
    this.konva.interactionRect.listening(isMoving);

    if (this.objects.size === 0) {
      // If the layer is totally empty, reset the cache and bail out.
      this.konva.transformer.nodes([]);
      if (this.konva.objectGroup.isCached()) {
        this.konva.objectGroup.clearCache();
      }
    } else if (isSelected && selectedTool === 'transform') {
      // When the layer is selected and being moved, we should always cache it.
      // We should update the cache if we drew to the layer.
      if (!this.konva.objectGroup.isCached() || didDraw) {
        // this.konva.objectGroup.cache();
      }
      // Activate the transformer
      this.konva.transformer.nodes([this.konva.group]);
      this.konva.transformer.forceUpdate();
      this.konva.transformer.visible(true);
    } else if (selectedTool === 'move') {
      // When the layer is selected and being moved, we should always cache it.
      // We should update the cache if we drew to the layer.
      if (!this.konva.objectGroup.isCached() || didDraw) {
        // this.konva.objectGroup.cache();
      }
      // Activate the transformer
      this.konva.transformer.nodes([]);
      this.konva.transformer.forceUpdate();
      this.konva.transformer.visible(false);
    } else if (isSelected) {
      // If the layer is selected but not using the move tool, we don't want the layer to be listening.
      // The transformer also does not need to be active.
      this.konva.transformer.nodes([]);
      if (isDrawingTool(selectedTool)) {
        // We are using a drawing tool (brush, eraser, rect). These tools change the layer's rendered appearance, so we
        // should never be cached.
        if (this.konva.objectGroup.isCached()) {
          this.konva.objectGroup.clearCache();
        }
      } else {
        // We are using a non-drawing tool (move, view, bbox), so we should cache the layer.
        // We should update the cache if we drew to the layer.
        if (!this.konva.objectGroup.isCached() || didDraw) {
          // this.konva.objectGroup.cache();
        }
      }
    } else if (!isSelected) {
      // Unselected layers should not be listening
      // The transformer also does not need to be active.
      this.konva.transformer.nodes([]);
      // Update the layer's cache if it's not already cached or we drew to it.
      if (!this.konva.objectGroup.isCached() || didDraw) {
        // this.konva.objectGroup.cache();
      }
    }
  }

  renderBbox() {
    const isSelected = this.manager.stateApi.getIsSelected(this.id);
    const selectedTool = this.manager.stateApi.getToolState().selected;
    const hasBbox = this.bbox.width !== 0 && this.bbox.height !== 0;

    this.konva.bbox.visible(hasBbox);
    this.konva.interactionRect.visible(hasBbox);

    this.konva.bbox.setAttrs({
      x: this.bbox.x,
      y: this.bbox.y,
      width: this.bbox.width,
      height: this.bbox.height,
      scaleX: 1,
      scaleY: 1,
      strokeWidth: 1 / this.manager.stage.scaleX(),
    });
    this.konva.interactionRect.setAttrs({
      x: this.bbox.x,
      y: this.bbox.y,
      width: this.bbox.width,
      height: this.bbox.height,
      scaleX: 1,
      scaleY: 1,
    });
  }

  private _getBbox() {
    if (this.objects.size === 0) {
      this.bbox = CanvasLayer.DEFAULT_BBOX_RECT;
      this.renderBbox();
      return;
    }

    let needsPixelBbox = false;
    const rect = this.konva.objectGroup.getClientRect({ skipTransform: true });
    // console.log('rect', rect);
    // If there are no eraser strokes, we can use the client rect directly
    for (const obj of this.objects.values()) {
      if (obj instanceof CanvasEraserLine) {
        needsPixelBbox = true;
        break;
      }
    }

    if (!needsPixelBbox) {
      if (rect.width === 0 || rect.height === 0) {
        this.bbox = CanvasLayer.DEFAULT_BBOX_RECT;
      } else {
        this.bbox = rect;
      }
      this.renderBbox();
      return;
    }

    // We have eraser strokes - we must calculate the bbox using pixel data

    // const a = window.performance.now();
    const clone = this.konva.objectGroup.clone();
    // const b = window.performance.now();
    // console.log('cloned layer', b - a);
    // const c = window.performance.now();
    const canvas = clone.toCanvas();
    // const d = window.performance.now();
    // console.log('got canvas', d - c);
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return;
    }
    const imageData = ctx.getImageData(0, 0, rect.width, rect.height);
    // const e = window.performance.now();
    // console.log('got image data', e - d);
    this.manager.requestBbox(
      { buffer: imageData.data.buffer, width: imageData.width, height: imageData.height },
      (extents) => {
        // console.log('extents', extents);
        if (extents) {
          const { minX, minY, maxX, maxY } = extents;
          this.bbox = {
            x: rect.x + minX,
            y: rect.y + minY,
            width: maxX - minX,
            height: maxY - minY,
          };
        } else {
          this.bbox = CanvasLayer.DEFAULT_BBOX_RECT;
        }
        this.renderBbox();
        clone.destroy();
        // console.log('bbox', this.bbox);
      }
    );
    // console.log('transferred message', window.performance.now() - e);
  }
}
