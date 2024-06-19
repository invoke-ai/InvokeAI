import type { EntityToKonvaMap, EntityToKonvaMapping } from 'features/controlLayers/konva/entityToKonvaMap';
import {
  RASTER_LAYER_BRUSH_LINE_NAME,
  RASTER_LAYER_ERASER_LINE_NAME,
  RASTER_LAYER_IMAGE_NAME,
  RASTER_LAYER_NAME,
  RASTER_LAYER_OBJECT_GROUP_NAME,
  RASTER_LAYER_RECT_SHAPE_NAME,
} from 'features/controlLayers/konva/naming';
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
  map: EntityToKonvaMap,
  entity: LayerEntity,
  onPosChanged?: (arg: PosChangedArg, entityType: CanvasEntity['type']) => void
): EntityToKonvaMapping => {
  let mapping = map.getMapping(entity.id);
  if (mapping) {
    return mapping;
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
  map.stage.add(konvaLayer);
  mapping = map.addMapping(entity.id, konvaLayer, konvaObjectGroup);
  return mapping;
};

/**
 * Renders a regional guidance layer.
 * @param stage The konva stage
 * @param entity The regional guidance layer state
 * @param tool The current tool
 * @param onPosChanged Callback for when the layer's position changes
 */
export const renderLayer = async (
  map: EntityToKonvaMap,
  entity: LayerEntity,
  tool: Tool,
  onPosChanged?: (arg: PosChangedArg, entityType: CanvasEntity['type']) => void
) => {
  const mapping = getLayer(map, entity, onPosChanged);

  // Update the layer's position and listening state
  mapping.konvaLayer.setAttrs({
    listening: tool === 'move', // The layer only listens when using the move tool - otherwise the stage is handling mouse events
    x: Math.floor(entity.x),
    y: Math.floor(entity.y),
  });

  const objectIds = entity.objects.map(mapId);
  // Destroy any objects that are no longer in state
  for (const entry of mapping.getEntries()) {
    if (!objectIds.includes(entry.id)) {
      mapping.destroyEntry(entry.id);
    }
  }

  for (const obj of entity.objects) {
    if (obj.type === 'brush_line') {
      const entry = getBrushLine(mapping, obj, RASTER_LAYER_BRUSH_LINE_NAME);
      // Only update the points if they have changed.
      if (entry.konvaLine.points().length !== obj.points.length) {
        entry.konvaLine.points(obj.points);
      }
    } else if (obj.type === 'eraser_line') {
      const entry = getEraserLine(mapping, obj, RASTER_LAYER_ERASER_LINE_NAME);
      // Only update the points if they have changed.
      if (entry.konvaLine.points().length !== obj.points.length) {
        entry.konvaLine.points(obj.points);
      }
    } else if (obj.type === 'rect_shape') {
      getRectShape(mapping, obj, RASTER_LAYER_RECT_SHAPE_NAME);
    } else if (obj.type === 'image') {
      createImageObjectGroup({ mapping, obj, name: RASTER_LAYER_IMAGE_NAME });
    }
  }

  // Only update layer visibility if it has changed.
  if (mapping.konvaLayer.visible() !== entity.isEnabled) {
    mapping.konvaLayer.visible(entity.isEnabled);
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

  mapping.konvaObjectGroup.opacity(entity.opacity);
};

export const renderLayers = (
  map: EntityToKonvaMap,
  entities: LayerEntity[],
  tool: Tool,
  onPosChanged?: (arg: PosChangedArg, entityType: CanvasEntity['type']) => void
): void => {
  // Destroy nonexistent layers
  for (const mapping of map.getMappings()) {
    if (!entities.find((l) => l.id === mapping.id)) {
      map.destroyMapping(mapping.id);
    }
  }
  for (const layer of entities) {
    renderLayer(map, layer, tool, onPosChanged);
  }
};
