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
  static GROUP_NAME = `${CanvasLayer.NAME_PREFIX}_group`;
  static OBJECT_GROUP_NAME = `${CanvasLayer.NAME_PREFIX}_object-group`;

  private drawingBuffer: BrushLine | EraserLine | RectShape | null;
  private state: LayerEntity;

  id: string;
  manager: CanvasManager;

  konva: {
    layer: Konva.Layer;
    bbox: Konva.Rect;
    group: Konva.Group;
    objectGroup: Konva.Group;
    transformer: Konva.Transformer;
  };
  objects: Map<string, CanvasBrushLine | CanvasEraserLine | CanvasRect | CanvasImage>;
  bbox: Rect | null;

  getBbox = debounce(this._getBbox, 300);

  constructor(state: LayerEntity, manager: CanvasManager) {
    this.id = state.id;
    this.manager = manager;
    this.konva = {
      layer: new Konva.Layer({ name: CanvasLayer.LAYER_NAME, listening: false }),
      group: new Konva.Group({ name: CanvasLayer.GROUP_NAME, listening: true }),
      bbox: new Konva.Rect({
        listening: true,
        stroke: 'hsl(200deg 76% 59%)', // invokeBlue.400
      }),
      objectGroup: new Konva.Group({ name: CanvasLayer.OBJECT_GROUP_NAME, listening: false }),
      transformer: new Konva.Transformer({
        name: CanvasLayer.TRANSFORMER_NAME,
        shouldOverdrawWholeArea: true,
        draggable: true,
        dragDistance: 0,
        enabledAnchors: ['top-left', 'top-right', 'bottom-left', 'bottom-right'],
        rotateEnabled: false,
        flipEnabled: false,
      }),
    };

    this.konva.group.add(this.konva.objectGroup);
    this.konva.group.add(this.konva.bbox);
    this.konva.layer.add(this.konva.group);

    this.konva.transformer.on('transformend', () => {
      this.manager.stateApi.onScaleChanged(
        {
          id: this.id,
          scale: this.konva.group.scaleX(),
          position: { x: this.konva.group.x(), y: this.konva.group.y() },
        },
        'layer'
      );
    });
    this.konva.transformer.on('dragend', () => {
      this.manager.stateApi.onPosChanged(
        { id: this.id, position: { x: this.konva.group.x(), y: this.konva.group.y() } },
        'layer'
      );
    });
    this.konva.layer.add(this.konva.transformer);

    this.objects = new Map();
    this.drawingBuffer = null;
    this.state = state;
    this.bbox = null;
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
      this.getBbox();
    }

    this.konva.layer.visible(true);
    this.konva.group.opacity(this.state.opacity);
    const isSelected = this.manager.stateApi.getIsSelected(this.id);
    const selectedTool = this.manager.stateApi.getToolState().selected;

    if (this.objects.size === 0) {
      // If the layer is totally empty, reset the cache and bail out.
      this.konva.layer.listening(false);
      this.konva.transformer.nodes([]);
      if (this.konva.group.isCached()) {
        this.konva.group.clearCache();
      }
    } else if (isSelected && selectedTool === 'move') {
      // When the layer is selected and being moved, we should always cache it.
      // We should update the cache if we drew to the layer.
      if (!this.konva.group.isCached() || didDraw) {
        // this.konva.group.cache();
      }
      // Activate the transformer
      this.konva.layer.listening(true);
      this.konva.transformer.nodes([this.konva.group]);
      this.konva.transformer.forceUpdate();
    } else if (isSelected && selectedTool !== 'move') {
      // If the layer is selected but not using the move tool, we don't want the layer to be listening.
      this.konva.layer.listening(false);
      // The transformer also does not need to be active.
      this.konva.transformer.nodes([]);
      if (isDrawingTool(selectedTool)) {
        // We are using a drawing tool (brush, eraser, rect). These tools change the layer's rendered appearance, so we
        // should never be cached.
        if (this.konva.group.isCached()) {
          this.konva.group.clearCache();
        }
      } else {
        // We are using a non-drawing tool (move, view, bbox), so we should cache the layer.
        // We should update the cache if we drew to the layer.
        if (!this.konva.group.isCached() || didDraw) {
          // this.konva.group.cache();
        }
      }
    } else if (!isSelected) {
      // Unselected layers should not be listening
      this.konva.layer.listening(false);
      // The transformer also does not need to be active.
      this.konva.transformer.nodes([]);
      // Update the layer's cache if it's not already cached or we drew to it.
      if (!this.konva.group.isCached() || didDraw) {
        // this.konva.group.cache();
      }
    }
  }

  renderBbox() {
    if (!this.bbox) {
      this.konva.bbox.visible(false);
      return;
    }
    this.konva.bbox.visible(true);
    this.konva.bbox.strokeWidth(1 / this.manager.stage.scaleX());
    this.konva.bbox.setAttrs(this.bbox);
  }

  private _getBbox() {
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
        this.bbox = null;
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
          this.bbox = {
            x: extents.minX + rect.x - Math.floor(this.konva.layer.x()),
            y: extents.minY + rect.y - Math.floor(this.konva.layer.y()),
            width: extents.maxX - extents.minX,
            height: extents.maxY - extents.minY,
          };
        } else {
          this.bbox = null;
        }
        this.renderBbox();
        clone.destroy();
        // console.log('bbox', this.bbox);
      }
    );
    // console.log('transferred message', window.performance.now() - e);
  }
}
