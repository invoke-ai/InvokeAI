import {
  RASTER_LAYER_BRUSH_LINE_NAME,
  RASTER_LAYER_ERASER_LINE_NAME,
  RASTER_LAYER_IMAGE_NAME,
  RASTER_LAYER_NAME,
  RASTER_LAYER_OBJECT_GROUP_NAME,
  RASTER_LAYER_RECT_SHAPE_NAME,
} from 'features/controlLayers/konva/naming';
import type { KonvaEntityAdapter, KonvaNodeManager } from 'features/controlLayers/konva/nodeManager';
import {
  createImageObjectGroup,
  createObjectGroup,
  getBrushLine,
  getEraserLine,
  getRectShape,
} from 'features/controlLayers/konva/renderers/objects';
import { mapId } from 'features/controlLayers/konva/util';
import type { CanvasEntity, LayerEntity, PosChangedArg, Tool } from 'features/controlLayers/store/types';
import Konva from 'konva';

/**
 * Logic for creating and rendering raster layers.
 */

/**
 * Creates a raster layer.
 * @param stage The konva stage
 * @param entity The raster layer state
 * @param onPosChanged Callback for when the layer's position changes
 */
const getLayer = (
  manager: KonvaNodeManager,
  entity: LayerEntity,
  onPosChanged?: (arg: PosChangedArg, entityType: CanvasEntity['type']) => void
): KonvaEntityAdapter => {
  const adapter = manager.get(entity.id);
  if (adapter) {
    return adapter;
  }
  // This layer hasn't been added to the konva state yet
  const konvaLayer = new Konva.Layer({
    id: entity.id,
    name: RASTER_LAYER_NAME,
    draggable: true,
    dragDistance: 0,
  });

  // When a drag on the layer finishes, update the layer's position in state. During the drag, konva handles changing
  // the position - we do not need to call this on the `dragmove` event.
  if (onPosChanged) {
    konvaLayer.on('dragend', function (e) {
      onPosChanged({ id: entity.id, x: Math.floor(e.target.x()), y: Math.floor(e.target.y()) }, 'layer');
    });
  }

  const konvaObjectGroup = createObjectGroup(konvaLayer, RASTER_LAYER_OBJECT_GROUP_NAME);
  return manager.add(entity, konvaLayer, konvaObjectGroup);
};

/**
 * Renders a regional guidance layer.
 * @param stage The konva stage
 * @param entity The regional guidance layer state
 * @param tool The current tool
 * @param onPosChanged Callback for when the layer's position changes
 */
export const renderLayer = async (
  manager: KonvaNodeManager,
  entity: LayerEntity,
  tool: Tool,
  onPosChanged?: (arg: PosChangedArg, entityType: CanvasEntity['type']) => void
) => {
  const adapter = getLayer(manager, entity, onPosChanged);

  // Update the layer's position and listening state
  adapter.konvaLayer.setAttrs({
    listening: tool === 'move', // The layer only listens when using the move tool - otherwise the stage is handling mouse events
    x: Math.floor(entity.x),
    y: Math.floor(entity.y),
  });

  const objectIds = entity.objects.map(mapId);
  // Destroy any objects that are no longer in state
  for (const objectRecord of adapter.getAll()) {
    if (!objectIds.includes(objectRecord.id)) {
      adapter.destroy(objectRecord.id);
    }
  }

  for (const obj of entity.objects) {
    if (obj.type === 'brush_line') {
      const objectRecord = getBrushLine(adapter, obj, RASTER_LAYER_BRUSH_LINE_NAME);
      // Only update the points if they have changed.
      if (objectRecord.konvaLine.points().length !== obj.points.length) {
        objectRecord.konvaLine.points(obj.points);
      }
    } else if (obj.type === 'eraser_line') {
      const objectRecord = getEraserLine(adapter, obj, RASTER_LAYER_ERASER_LINE_NAME);
      // Only update the points if they have changed.
      if (objectRecord.konvaLine.points().length !== obj.points.length) {
        objectRecord.konvaLine.points(obj.points);
      }
    } else if (obj.type === 'rect_shape') {
      getRectShape(adapter, obj, RASTER_LAYER_RECT_SHAPE_NAME);
    } else if (obj.type === 'image') {
      createImageObjectGroup({ adapter, obj, name: RASTER_LAYER_IMAGE_NAME });
    }
  }

  // Only update layer visibility if it has changed.
  if (adapter.konvaLayer.visible() !== entity.isEnabled) {
    adapter.konvaLayer.visible(entity.isEnabled);
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

  adapter.konvaObjectGroup.opacity(entity.opacity);
};

export const renderLayers = (
  manager: KonvaNodeManager,
  entities: LayerEntity[],
  tool: Tool,
  onPosChanged?: (arg: PosChangedArg, entityType: CanvasEntity['type']) => void
): void => {
  // Destroy nonexistent layers
  for (const adapter of manager.getAll('layer')) {
    if (!entities.find((l) => l.id === adapter.id)) {
      manager.destroy(adapter.id);
    }
  }
  for (const entity of entities) {
    renderLayer(manager, entity, tool, onPosChanged);
  }
};
