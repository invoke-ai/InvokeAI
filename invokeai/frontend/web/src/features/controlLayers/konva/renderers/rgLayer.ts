import { rgbColorToString } from 'features/canvas/util/colorToString';
import {
  COMPOSITING_RECT_NAME,
  getObjectGroupId,
  RG_LAYER_NAME,
  RG_LAYER_OBJECT_GROUP_NAME,
} from 'features/controlLayers/konva/naming';
import { getLayerBboxFast } from 'features/controlLayers/konva/renderers/bbox';
import { createBrushLine, createEraserLine, createRectShape } from 'features/controlLayers/konva/renderers/objects';
import { getScaledFlooredCursorPosition, mapId, selectVectorMaskObjects } from 'features/controlLayers/konva/util';
import type { RegionalGuidanceLayer, Tool } from 'features/controlLayers/store/types';
import Konva from 'konva';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

/**
 * Logic for creating and rendering regional guidance layers.
 *
 * Some special handling is needed to render layer opacity correctly using a "compositing rect". See the comments
 * in `renderRGLayer`.
 */

/**
 * Creates the "compositing rect" for a regional guidance layer.
 * @param konvaLayer The konva layer
 */
const createCompositingRect = (konvaLayer: Konva.Layer): Konva.Rect => {
  const compositingRect = new Konva.Rect({ name: COMPOSITING_RECT_NAME, listening: false });
  konvaLayer.add(compositingRect);
  return compositingRect;
};

/**
 * Creates a regional guidance layer.
 * @param stage The konva stage
 * @param layerState The regional guidance layer state
 * @param onLayerPosChanged Callback for when the layer's position changes
 */
const createRGLayer = (
  stage: Konva.Stage,
  layerState: RegionalGuidanceLayer,
  onLayerPosChanged?: (layerId: string, x: number, y: number) => void
): Konva.Layer => {
  // This layer hasn't been added to the konva state yet
  const konvaLayer = new Konva.Layer({
    id: layerState.id,
    name: RG_LAYER_NAME,
    draggable: true,
    dragDistance: 0,
  });

  // When a drag on the layer finishes, update the layer's position in state. During the drag, konva handles changing
  // the position - we do not need to call this on the `dragmove` event.
  if (onLayerPosChanged) {
    konvaLayer.on('dragend', function (e) {
      onLayerPosChanged(layerState.id, Math.floor(e.target.x()), Math.floor(e.target.y()));
    });
  }

  // The dragBoundFunc limits how far the layer can be dragged
  konvaLayer.dragBoundFunc(function (pos) {
    const cursorPos = getScaledFlooredCursorPosition(stage);
    if (!cursorPos) {
      return this.getAbsolutePosition();
    }
    // Prevent the user from dragging the layer out of the stage bounds by constaining the cursor position to the stage bounds
    if (
      cursorPos.x < 0 ||
      cursorPos.x > stage.width() / stage.scaleX() ||
      cursorPos.y < 0 ||
      cursorPos.y > stage.height() / stage.scaleY()
    ) {
      return this.getAbsolutePosition();
    }
    return pos;
  });

  // The object group holds all of the layer's objects (e.g. lines and rects)
  const konvaObjectGroup = new Konva.Group({
    id: getObjectGroupId(layerState.id, uuidv4()),
    name: RG_LAYER_OBJECT_GROUP_NAME,
    listening: false,
  });
  konvaLayer.add(konvaObjectGroup);

  stage.add(konvaLayer);

  return konvaLayer;
};

/**
 * Renders a raster layer.
 * @param stage The konva stage
 * @param layerState The regional guidance layer state
 * @param globalMaskLayerOpacity The global mask layer opacity
 * @param tool The current tool
 * @param onLayerPosChanged Callback for when the layer's position changes
 */
export const renderRGLayer = (
  stage: Konva.Stage,
  layerState: RegionalGuidanceLayer,
  globalMaskLayerOpacity: number,
  tool: Tool,
  onLayerPosChanged?: (layerId: string, x: number, y: number) => void
): void => {
  const konvaLayer =
    stage.findOne<Konva.Layer>(`#${layerState.id}`) ?? createRGLayer(stage, layerState, onLayerPosChanged);

  // Update the layer's position and listening state
  konvaLayer.setAttrs({
    listening: tool === 'move', // The layer only listens when using the move tool - otherwise the stage is handling mouse events
    x: Math.floor(layerState.x),
    y: Math.floor(layerState.y),
  });

  // Convert the color to a string, stripping the alpha - the object group will handle opacity.
  const rgbColor = rgbColorToString(layerState.previewColor);

  const konvaObjectGroup = konvaLayer.findOne<Konva.Group>(`.${RG_LAYER_OBJECT_GROUP_NAME}`);
  assert(konvaObjectGroup, `Object group not found for layer ${layerState.id}`);

  // We use caching to handle "global" layer opacity, but caching is expensive and we should only do it when required.
  let groupNeedsCache = false;

  const objectIds = layerState.objects.map(mapId);
  // Destroy any objects that are no longer in the redux state
  for (const objectNode of konvaObjectGroup.find(selectVectorMaskObjects)) {
    if (!objectIds.includes(objectNode.id())) {
      objectNode.destroy();
      groupNeedsCache = true;
    }
  }

  for (const obj of layerState.objects) {
    if (obj.type === 'brush_line') {
      const konvaBrushLine = stage.findOne<Konva.Line>(`#${obj.id}`) ?? createBrushLine(obj, konvaObjectGroup);

      // Only update the points if they have changed. The point values are never mutated, they are only added to the
      // array, so checking the length is sufficient to determine if we need to re-cache.
      if (konvaBrushLine.points().length !== obj.points.length) {
        konvaBrushLine.points(obj.points);
        groupNeedsCache = true;
      }
      // Only update the color if it has changed.
      if (konvaBrushLine.stroke() !== rgbColor) {
        konvaBrushLine.stroke(rgbColor);
        groupNeedsCache = true;
      }
    } else if (obj.type === 'eraser_line') {
      const konvaEraserLine = stage.findOne<Konva.Line>(`#${obj.id}`) ?? createEraserLine(obj, konvaObjectGroup);

      // Only update the points if they have changed. The point values are never mutated, they are only added to the
      // array, so checking the length is sufficient to determine if we need to re-cache.
      if (konvaEraserLine.points().length !== obj.points.length) {
        konvaEraserLine.points(obj.points);
        groupNeedsCache = true;
      }
      // Only update the color if it has changed.
      if (konvaEraserLine.stroke() !== rgbColor) {
        konvaEraserLine.stroke(rgbColor);
        groupNeedsCache = true;
      }
    } else if (obj.type === 'rect_shape') {
      const konvaRectShape = stage.findOne<Konva.Rect>(`#${obj.id}`) ?? createRectShape(obj, konvaObjectGroup);

      // Only update the color if it has changed.
      if (konvaRectShape.fill() !== rgbColor) {
        konvaRectShape.fill(rgbColor);
        groupNeedsCache = true;
      }
    }
  }

  // Only update layer visibility if it has changed.
  if (konvaLayer.visible() !== layerState.isEnabled) {
    konvaLayer.visible(layerState.isEnabled);
    groupNeedsCache = true;
  }

  if (konvaObjectGroup.getChildren().length === 0) {
    // No objects - clear the cache to reset the previous pixel data
    konvaObjectGroup.clearCache();
    return;
  }

  const compositingRect =
    konvaLayer.findOne<Konva.Rect>(`.${COMPOSITING_RECT_NAME}`) ?? createCompositingRect(konvaLayer);

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
  if (layerState.isSelected && tool !== 'move') {
    // We must clear the cache first so Konva will re-draw the group with the new compositing rect
    if (konvaObjectGroup.isCached()) {
      konvaObjectGroup.clearCache();
    }
    // The user is allowed to reduce mask opacity to 0, but we need the opacity for the compositing rect to work
    konvaObjectGroup.opacity(1);

    compositingRect.setAttrs({
      // The rect should be the size of the layer - use the fast method if we don't have a pixel-perfect bbox already
      ...(!layerState.bboxNeedsUpdate && layerState.bbox ? layerState.bbox : getLayerBboxFast(konvaLayer)),
      fill: rgbColor,
      opacity: globalMaskLayerOpacity,
      // Draw this rect only where there are non-transparent pixels under it (e.g. the mask shapes)
      globalCompositeOperation: 'source-in',
      visible: true,
      // This rect must always be on top of all other shapes
      zIndex: konvaObjectGroup.getChildren().length,
    });
  } else {
    // The compositing rect should only be shown when the layer is selected.
    compositingRect.visible(false);
    // Cache only if needed - or if we are on this code path and _don't_ have a cache
    if (groupNeedsCache || !konvaObjectGroup.isCached()) {
      konvaObjectGroup.cache();
    }
    // Updating group opacity does not require re-caching
    konvaObjectGroup.opacity(globalMaskLayerOpacity);
  }
};
