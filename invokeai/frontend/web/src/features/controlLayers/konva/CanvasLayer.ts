import { CanvasBrushLine } from 'features/controlLayers/konva/CanvasBrushLine';
import { CanvasEraserLine } from 'features/controlLayers/konva/CanvasEraserLine';
import { CanvasImage } from 'features/controlLayers/konva/CanvasImage';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasRect } from 'features/controlLayers/konva/CanvasRect';
import { getObjectGroupId } from 'features/controlLayers/konva/naming';
import { mapId } from 'features/controlLayers/konva/util';
import type { BrushLine, EraserLine, LayerEntity } from 'features/controlLayers/store/types';
import { isDrawingTool } from 'features/controlLayers/store/types';
import Konva from 'konva';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

export class CanvasLayer {
  id: string;
  manager: CanvasManager;
  layer: Konva.Layer;
  group: Konva.Group;
  objectsGroup: Konva.Group;
  transformer: Konva.Transformer;
  objects: Map<string, CanvasBrushLine | CanvasEraserLine | CanvasRect | CanvasImage>;
  private drawingBuffer: BrushLine | EraserLine | null;
  private prevLayerState: LayerEntity;

  constructor(entity: LayerEntity, manager: CanvasManager) {
    this.id = entity.id;
    this.manager = manager;
    this.layer = new Konva.Layer({
      id: entity.id,
      listening: false,
    });

    this.group = new Konva.Group({
      id: getObjectGroupId(this.layer.id(), uuidv4()),
      listening: false,
    });
    this.objectsGroup = new Konva.Group({});
    this.group.add(this.objectsGroup);
    this.layer.add(this.group);

    this.transformer = new Konva.Transformer({
      shouldOverdrawWholeArea: true,
      draggable: true,
      dragDistance: 0,
      enabledAnchors: ['top-left', 'top-right', 'bottom-left', 'bottom-right'],
      rotateEnabled: false,
      flipEnabled: false,
    });
    this.transformer.on('transformend', () => {
      this.manager.stateApi.onScaleChanged(
        { id: this.id, scale: this.group.scaleX(), x: this.group.x(), y: this.group.y() },
        'layer'
      );
    });
    this.transformer.on('dragend', () => {
      this.manager.stateApi.onPosChanged({ id: this.id, x: this.group.x(), y: this.group.y() }, 'layer');
    });
    this.layer.add(this.transformer);

    this.objects = new Map();
    this.drawingBuffer = null;
    this.prevLayerState = entity;
  }

  destroy(): void {
    this.layer.destroy();
  }

  getDrawingBuffer() {
    return this.drawingBuffer;
  }

  async setDrawingBuffer(obj: BrushLine | EraserLine | null) {
    if (obj) {
      this.drawingBuffer = obj;
      await this.renderObject(this.drawingBuffer, true);
      this.updateGroup(true, this.prevLayerState);
    } else {
      this.drawingBuffer = null;
    }
  }

  finalizeDrawingBuffer() {
    if (!this.drawingBuffer) {
      return;
    }
    if (this.drawingBuffer.type === 'brush_line') {
      this.manager.stateApi.onBrushLineAdded2({ id: this.id, brushLine: this.drawingBuffer }, 'layer');
    } else if (this.drawingBuffer.type === 'eraser_line') {
      this.manager.stateApi.onEraserLineAdded2({ id: this.id, eraserLine: this.drawingBuffer }, 'layer');
    }
    this.setDrawingBuffer(null);
  }

  async render(layerState: LayerEntity) {
    // Update the layer's position and listening state
    this.group.setAttrs({
      x: layerState.x,
      y: layerState.y,
      scaleX: 1,
      scaleY: 1,
    });

    let didDraw = false;

    const objectIds = layerState.objects.map(mapId);
    // Destroy any objects that are no longer in state
    for (const object of this.objects.values()) {
      if (!objectIds.includes(object.id) && object.id !== this.drawingBuffer?.id) {
        this.objects.delete(object.id);
        object.destroy();
        didDraw = true;
      }
    }

    for (const obj of layerState.objects) {
      didDraw = await this.renderObject(obj);
    }

    if (this.drawingBuffer) {
      didDraw = await this.renderObject(this.drawingBuffer);
    }

    // Only update layer visibility if it has changed.
    if (this.layer.visible() !== layerState.isEnabled) {
      this.layer.visible(layerState.isEnabled);
    }

    this.group.opacity(layerState.opacity);

    // The layer only listens when using the move tool - otherwise the stage is handling mouse events
    this.updateGroup(didDraw, this.prevLayerState);

    this.prevLayerState = layerState;
  }

  private async renderObject(obj: LayerEntity['objects'][number], force = false): Promise<boolean> {
    if (obj.type === 'brush_line') {
      let brushLine = this.objects.get(obj.id);
      assert(brushLine instanceof CanvasBrushLine || brushLine === undefined);

      if (!brushLine) {
        brushLine = new CanvasBrushLine(obj);
        this.objects.set(brushLine.id, brushLine);
        this.objectsGroup.add(brushLine.konvaLineGroup);
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
        this.objectsGroup.add(eraserLine.konvaLineGroup);
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
        this.objectsGroup.add(rect.konvaRect);
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
        image = await new CanvasImage(obj, {
          onLoad: () => {
            this.updateGroup(true, this.prevLayerState);
          },
        });
        this.objects.set(image.id, image);
        this.objectsGroup.add(image.konvaImageGroup);
        await image.updateImageSource(obj.image.name);
      } else {
        if (await image.update(obj, force)) {
          return true;
        }
      }
    }

    return false;
  }

  updateGroup(didDraw: boolean, _: LayerEntity) {
    const isSelected = this.manager.stateApi.getIsSelected(this.id);
    const selectedTool = this.manager.stateApi.getToolState().selected;

    if (this.objects.size === 0) {
      // If the layer is totally empty, reset the cache and bail out.
      this.layer.listening(false);
      this.transformer.nodes([]);
      if (this.group.isCached()) {
        this.group.clearCache();
      }
      return;
    }

    if (isSelected && selectedTool === 'move') {
      // When the layer is selected and being moved, we should always cache it.
      // We should update the cache if we drew to the layer.
      if (!this.group.isCached() || didDraw) {
        this.group.cache();
      }
      // Activate the transformer
      this.layer.listening(true);
      this.transformer.nodes([this.group]);
      this.transformer.forceUpdate();
      return;
    }

    if (isSelected && selectedTool !== 'move') {
      // If the layer is selected but not using the move tool, we don't want the layer to be listening.
      this.layer.listening(false);
      // The transformer also does not need to be active.
      this.transformer.nodes([]);
      if (isDrawingTool(selectedTool)) {
        // We are using a drawing tool (brush, eraser, rect). These tools change the layer's rendered appearance, so we
        // should never be cached.
        if (this.group.isCached()) {
          this.group.clearCache();
        }
      } else {
        // We are using a non-drawing tool (move, view, bbox), so we should cache the layer.
        // We should update the cache if we drew to the layer.
        if (!this.group.isCached() || didDraw) {
          this.group.cache();
        }
      }
      return;
    }

    if (!isSelected) {
      // Unselected layers should not be listening
      this.layer.listening(false);
      // The transformer also does not need to be active.
      this.transformer.nodes([]);
      // Update the layer's cache if it's not already cached or we drew to it.
      if (!this.group.isCached() || didDraw) {
        this.group.cache();
      }

      return;
    }
  }
}
