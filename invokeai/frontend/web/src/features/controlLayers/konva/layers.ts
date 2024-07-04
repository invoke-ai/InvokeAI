import { getObjectGroupId } from 'features/controlLayers/konva/naming';
import type { KonvaNodeManager } from 'features/controlLayers/konva/nodeManager';
import { KonvaBrushLine, KonvaEraserLine, KonvaImage, KonvaRect } from 'features/controlLayers/konva/objects';
import { mapId } from 'features/controlLayers/konva/util';
import { isDrawingTool, type LayerEntity } from 'features/controlLayers/store/types';
import Konva from 'konva';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

export class CanvasLayer {
  id: string;
  manager: KonvaNodeManager;
  layer: Konva.Layer;
  group: Konva.Group;
  transformer: Konva.Transformer;
  objects: Map<string, KonvaBrushLine | KonvaEraserLine | KonvaRect | KonvaImage>;

  constructor(entity: LayerEntity, manager: KonvaNodeManager) {
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
  }

  destroy(): void {
    this.layer.destroy();
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
      if (!objectIds.includes(object.id)) {
        this.objects.delete(object.id);
        object.destroy();
        didDraw = true;
      }
    }

    for (const obj of layerState.objects) {
      if (obj.type === 'brush_line') {
        let brushLine = this.objects.get(obj.id);
        assert(brushLine instanceof KonvaBrushLine || brushLine === undefined);

        if (!brushLine) {
          brushLine = new KonvaBrushLine(obj);
          this.objects.set(brushLine.id, brushLine);
          this.group.add(brushLine.konvaLineGroup);
          didDraw = true;
        } else {
          if (brushLine.update(obj)) {
            didDraw = true;
          }
        }
      } else if (obj.type === 'eraser_line') {
        let eraserLine = this.objects.get(obj.id);
        assert(eraserLine instanceof KonvaEraserLine || eraserLine === undefined);

        if (!eraserLine) {
          eraserLine = new KonvaEraserLine(obj);
          this.objects.set(eraserLine.id, eraserLine);
          this.group.add(eraserLine.konvaLineGroup);
          didDraw = true;
        } else {
          if (eraserLine.update(obj)) {
            didDraw = true;
          }
        }
      } else if (obj.type === 'rect_shape') {
        let rect = this.objects.get(obj.id);
        assert(rect instanceof KonvaRect || rect === undefined);

        if (!rect) {
          rect = new KonvaRect(obj);
          this.objects.set(rect.id, rect);
          this.group.add(rect.konvaRect);
          didDraw = true;
        } else {
          if (rect.update(obj)) {
            didDraw = true;
          }
        }
      } else if (obj.type === 'image') {
        let image = this.objects.get(obj.id);
        assert(image instanceof KonvaImage || image === undefined);

        if (!image) {
          image = await new KonvaImage(obj, {
            onLoad: () => {
              this.updateGroup(true);
            },
          });
          this.objects.set(image.id, image);
          this.group.add(image.konvaImageGroup);
          await image.updateImageSource(obj.image.name);
        } else {
          if (await image.update(obj)) {
            didDraw = true;
          }
        }
      }
    }

    // Only update layer visibility if it has changed.
    if (this.layer.visible() !== layerState.isEnabled) {
      this.layer.visible(layerState.isEnabled);
    }

    this.group.opacity(layerState.opacity);

    // The layer only listens when using the move tool - otherwise the stage is handling mouse events
    this.updateGroup(didDraw);
  }

  updateGroup(didDraw: boolean) {
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
