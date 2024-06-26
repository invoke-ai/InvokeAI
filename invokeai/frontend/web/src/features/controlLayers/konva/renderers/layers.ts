import { getObjectGroupId } from 'features/controlLayers/konva/naming';
import type { StateApi } from 'features/controlLayers/konva/nodeManager';
import { KonvaBrushLine, KonvaEraserLine, KonvaImage, KonvaRect } from 'features/controlLayers/konva/renderers/objects';
import { mapId } from 'features/controlLayers/konva/util';
import type { LayerEntity, Tool } from 'features/controlLayers/store/types';
import Konva from 'konva';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

export class CanvasLayer {
  id: string;
  konvaLayer: Konva.Layer;
  konvaObjectGroup: Konva.Group;
  objects: Map<string, KonvaBrushLine | KonvaEraserLine | KonvaRect | KonvaImage>;

  constructor(entity: LayerEntity, onPosChanged: StateApi['onPosChanged']) {
    this.id = entity.id;

    this.konvaLayer = new Konva.Layer({
      id: entity.id,
      draggable: true,
      dragDistance: 0,
    });

    // When a drag on the layer finishes, update the layer's position in state. During the drag, konva handles changing
    // the position - we do not need to call this on the `dragmove` event.
    this.konvaLayer.on('dragend', function (e) {
      onPosChanged({ id: entity.id, x: Math.floor(e.target.x()), y: Math.floor(e.target.y()) }, 'layer');
    });
    const konvaObjectGroup = new Konva.Group({
      id: getObjectGroupId(this.konvaLayer.id(), uuidv4()),
      listening: false,
    });
    this.konvaObjectGroup = konvaObjectGroup;
    this.konvaLayer.add(this.konvaObjectGroup);
    this.objects = new Map();
  }

  destroy(): void {
    this.konvaLayer.destroy();
  }

  async render(layerState: LayerEntity, selectedTool: Tool) {
    // Update the layer's position and listening state
    this.konvaLayer.setAttrs({
      listening: selectedTool === 'move', // The layer only listens when using the move tool - otherwise the stage is handling mouse events
      x: Math.floor(layerState.x),
      y: Math.floor(layerState.y),
    });

    const objectIds = layerState.objects.map(mapId);
    // Destroy any objects that are no longer in state
    for (const object of this.objects.values()) {
      if (!objectIds.includes(object.id)) {
        object.destroy();
      }
    }

    for (const obj of layerState.objects) {
      if (obj.type === 'brush_line') {
        let brushLine = this.objects.get(obj.id);
        assert(brushLine instanceof KonvaBrushLine || brushLine === undefined);

        if (!brushLine) {
          brushLine = new KonvaBrushLine({ brushLine: obj });
          this.objects.set(brushLine.id, brushLine);
          this.konvaLayer.add(brushLine.konvaLineGroup);
        }
        if (obj.points.length !== brushLine.konvaLine.points().length) {
          brushLine.konvaLine.points(obj.points);
        }
      } else if (obj.type === 'eraser_line') {
        let eraserLine = this.objects.get(obj.id);
        assert(eraserLine instanceof KonvaEraserLine || eraserLine === undefined);

        if (!eraserLine) {
          eraserLine = new KonvaEraserLine({ eraserLine: obj });
          this.objects.set(eraserLine.id, eraserLine);
          this.konvaLayer.add(eraserLine.konvaLineGroup);
        }
        if (obj.points.length !== eraserLine.konvaLine.points().length) {
          eraserLine.konvaLine.points(obj.points);
        }
      } else if (obj.type === 'rect_shape') {
        let rect = this.objects.get(obj.id);
        assert(rect instanceof KonvaRect || rect === undefined);

        if (!rect) {
          rect = new KonvaRect({ rectShape: obj });
          this.objects.set(rect.id, rect);
          this.konvaLayer.add(rect.konvaRect);
        }
      } else if (obj.type === 'image') {
        let image = this.objects.get(obj.id);
        assert(image instanceof KonvaImage || image === undefined);

        if (!image) {
          image = await new KonvaImage({ imageObject: obj });
          this.objects.set(image.id, image);
          this.konvaLayer.add(image.konvaImageGroup);
        }
        if (image.imageName !== obj.image.name) {
          image.updateImageSource(obj.image.name);
        }
      }
    }

    // Only update layer visibility if it has changed.
    if (this.konvaLayer.visible() !== layerState.isEnabled) {
      this.konvaLayer.visible(layerState.isEnabled);
    }

    // const bboxRect = konvaLayer.findOne<Konva.Rect>(`.${LAYER_BBOX_NAME}`) ?? createBboxRect(layerState, konvaLayer);
    // if (layerState.bbox) {
    //   const active = !layerState.bboxNeedsUpdate && layerState.isSelected && tool === 'move';
    //   bboxRect.setAttrs({
    //     visible: active,
    //     listening: active,
    //     x: layerState.bbox.x,
    //     y: layerState.bbox.y,
    //     width: layerState.bbox.width,
    //     height: layerState.bbox.height,
    //     stroke: layerState.isSelected ? BBOX_SELECTED_STROKE : '',
    //     strokeWidth: 1 / stage.scaleX(),
    //   });
    // } else {
    //   bboxRect.visible(false);
    // }
    this.konvaObjectGroup.opacity(layerState.opacity);
  }
}
