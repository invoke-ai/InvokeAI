import { rgbColorToString } from 'common/util/colorCodeTransformers';
import {
  COMPOSITING_RECT_NAME,
  INPAINT_MASK_LAYER_BRUSH_LINE_NAME,
  INPAINT_MASK_LAYER_ERASER_LINE_NAME,
  INPAINT_MASK_LAYER_NAME,
  INPAINT_MASK_LAYER_OBJECT_GROUP_NAME,
  INPAINT_MASK_LAYER_RECT_SHAPE_NAME,
} from 'features/controlLayers/konva/naming';
import type { KonvaEntityAdapter, KonvaNodeManager } from 'features/controlLayers/konva/nodeManager';
import { getLayerBboxFast } from 'features/controlLayers/konva/renderers/bbox';
import {
  createObjectGroup,
  getBrushLine,
  getEraserLine,
  getRectShape,
} from 'features/controlLayers/konva/renderers/objects';
import { mapId } from 'features/controlLayers/konva/util';
import type { CanvasEntity, InpaintMaskEntity, PosChangedArg } from 'features/controlLayers/store/types';
import Konva from 'konva';

/**
 * Creates the "compositing rect" for the inpaint mask.
 * @param konvaLayer The konva layer
 */
const createCompositingRect = (konvaLayer: Konva.Layer): Konva.Rect => {
  const compositingRect = new Konva.Rect({ name: COMPOSITING_RECT_NAME, listening: false });
  konvaLayer.add(compositingRect);
  return compositingRect;
};

/**
 * Gets the singleton inpaint mask entity's konva nodes and entity adapter, creating them if they do not exist.
 * @param manager The konva node manager
 * @param entityState The inpaint mask entity state
 * @param onPosChanged Callback for when the position changes (e.g. the entity is dragged)
 * @returns The konva entity adapter for the inpaint mask
 */
const getInpaintMask = (
  manager: KonvaNodeManager,
  entityState: InpaintMaskEntity,
  onPosChanged?: (arg: PosChangedArg, entityType: CanvasEntity['type']) => void
): KonvaEntityAdapter => {
  const adapter = manager.get(entityState.id);
  if (adapter) {
    return adapter;
  }
  // This layer hasn't been added to the konva state yet
  const konvaLayer = new Konva.Layer({
    id: entityState.id,
    name: INPAINT_MASK_LAYER_NAME,
    draggable: true,
    dragDistance: 0,
  });

  // When a drag on the layer finishes, update the layer's position in state. During the drag, konva handles changing
  // the position - we do not need to call this on the `dragmove` event.
  if (onPosChanged) {
    konvaLayer.on('dragend', function (e) {
      onPosChanged({ id: entityState.id, x: Math.floor(e.target.x()), y: Math.floor(e.target.y()) }, 'inpaint_mask');
    });
  }

  const konvaObjectGroup = createObjectGroup(konvaLayer, INPAINT_MASK_LAYER_OBJECT_GROUP_NAME);
  return manager.add(entityState, konvaLayer, konvaObjectGroup);
};

/**
 * Gets a function to render the inpaint mask.
 * @param manager The konva node manager
 * @returns A function to render the inpaint mask
 */
export const getRenderInpaintMask = (manager: KonvaNodeManager) => {
  const { getInpaintMaskEntityState, getMaskOpacity, getToolState, getSelectedEntity, onPosChanged } = manager.stateApi;

  function renderInpaintMask(): void {
    const entity = getInpaintMaskEntityState();
    const globalMaskLayerOpacity = getMaskOpacity();
    const toolState = getToolState();
    const selectedEntity = getSelectedEntity();
    const adapter = getInpaintMask(manager, entity, onPosChanged);

    // Update the layer's position and listening state
    adapter.konvaLayer.setAttrs({
      listening: toolState.selected === 'move', // The layer only listens when using the move tool - otherwise the stage is handling mouse events
      x: Math.floor(entity.x),
      y: Math.floor(entity.y),
    });

    // Convert the color to a string, stripping the alpha - the object group will handle opacity.
    const rgbColor = rgbColorToString(entity.fill);

    // We use caching to handle "global" layer opacity, but caching is expensive and we should only do it when required.
    let groupNeedsCache = false;

    const objectIds = entity.objects.map(mapId);
    // Destroy any objects that are no longer in state
    for (const objectRecord of adapter.getAll()) {
      if (!objectIds.includes(objectRecord.id)) {
        adapter.destroy(objectRecord.id);
        groupNeedsCache = true;
      }
    }

    for (const obj of entity.objects) {
      if (obj.type === 'brush_line') {
        const objectRecord = getBrushLine(adapter, obj, INPAINT_MASK_LAYER_BRUSH_LINE_NAME);

        // Only update the points if they have changed. The point values are never mutated, they are only added to the
        // array, so checking the length is sufficient to determine if we need to re-cache.
        if (objectRecord.konvaLine.points().length !== obj.points.length) {
          objectRecord.konvaLine.points(obj.points);
          groupNeedsCache = true;
        }
        // Only update the color if it has changed.
        if (objectRecord.konvaLine.stroke() !== rgbColor) {
          objectRecord.konvaLine.stroke(rgbColor);
          groupNeedsCache = true;
        }
      } else if (obj.type === 'eraser_line') {
        const objectRecord = getEraserLine(adapter, obj, INPAINT_MASK_LAYER_ERASER_LINE_NAME);

        // Only update the points if they have changed. The point values are never mutated, they are only added to the
        // array, so checking the length is sufficient to determine if we need to re-cache.
        if (objectRecord.konvaLine.points().length !== obj.points.length) {
          objectRecord.konvaLine.points(obj.points);
          groupNeedsCache = true;
        }
        // Only update the color if it has changed.
        if (objectRecord.konvaLine.stroke() !== rgbColor) {
          objectRecord.konvaLine.stroke(rgbColor);
          groupNeedsCache = true;
        }
      } else if (obj.type === 'rect_shape') {
        const objectRecord = getRectShape(adapter, obj, INPAINT_MASK_LAYER_RECT_SHAPE_NAME);

        // Only update the color if it has changed.
        if (objectRecord.konvaRect.fill() !== rgbColor) {
          objectRecord.konvaRect.fill(rgbColor);
          groupNeedsCache = true;
        }
      }
    }

    // Only update layer visibility if it has changed.
    if (adapter.konvaLayer.visible() !== entity.isEnabled) {
      adapter.konvaLayer.visible(entity.isEnabled);
      groupNeedsCache = true;
    }

    if (adapter.konvaObjectGroup.getChildren().length === 0) {
      // No objects - clear the cache to reset the previous pixel data
      adapter.konvaObjectGroup.clearCache();
      return;
    }

    const compositingRect =
      adapter.konvaLayer.findOne<Konva.Rect>(`.${COMPOSITING_RECT_NAME}`) ?? createCompositingRect(adapter.konvaLayer);
    const isSelected = selectedEntity?.id === entity.id;

    /**
     * When the group is selected, we use a rect of the selected preview color, composited over the shapes. This allows
     * shapes to render as a "raster" layer with all pixels drawn at the same color and opacity.
     *
     * Without this special handling, each shape is drawn individually with the given opacity, atop the other shapes. The
     * effect is like if you have a Photoshop Group consisting of many shapes, each of which has the given opacity.
     * Overlapping shapes will have their colors blended together, and the final color is the result of all the shapes.
     *
     * Instead, with the special handling, the effect is as if you drew all the shapes at 100% opacity, flattened them to
     * a single raster image, and _then_ applied the 50% opacity.
     */
    if (isSelected && toolState.selected !== 'move') {
      // We must clear the cache first so Konva will re-draw the group with the new compositing rect
      if (adapter.konvaObjectGroup.isCached()) {
        adapter.konvaObjectGroup.clearCache();
      }
      // The user is allowed to reduce mask opacity to 0, but we need the opacity for the compositing rect to work
      adapter.konvaObjectGroup.opacity(1);

      compositingRect.setAttrs({
        // The rect should be the size of the layer - use the fast method if we don't have a pixel-perfect bbox already
        ...(!entity.bboxNeedsUpdate && entity.bbox ? entity.bbox : getLayerBboxFast(adapter.konvaLayer)),
        fill: rgbColor,
        opacity: globalMaskLayerOpacity,
        // Draw this rect only where there are non-transparent pixels under it (e.g. the mask shapes)
        globalCompositeOperation: 'source-in',
        visible: true,
        // This rect must always be on top of all other shapes
        zIndex: adapter.konvaObjectGroup.getChildren().length,
      });
    } else {
      // The compositing rect should only be shown when the layer is selected.
      compositingRect.visible(false);
      // Cache only if needed - or if we are on this code path and _don't_ have a cache
      if (groupNeedsCache || !adapter.konvaObjectGroup.isCached()) {
        adapter.konvaObjectGroup.cache();
      }
      // Updating group opacity does not require re-caching
      adapter.konvaObjectGroup.opacity(globalMaskLayerOpacity);
    }

    // const bboxRect =
    //   regionMap.konvaLayer.findOne<Konva.Rect>(`.${LAYER_BBOX_NAME}`) ?? createBboxRect(rg, regionMap.konvaLayer);

    // if (rg.bbox) {
    //   const active = !rg.bboxNeedsUpdate && isSelected && tool === 'move';
    //   bboxRect.setAttrs({
    //     visible: active,
    //     listening: active,
    //     x: rg.bbox.x,
    //     y: rg.bbox.y,
    //     width: rg.bbox.width,
    //     height: rg.bbox.height,
    //     stroke: isSelected ? BBOX_SELECTED_STROKE : '',
    //   });
    // } else {
    //   bboxRect.visible(false);
    // }
  }

  return renderInpaintMask;
};
