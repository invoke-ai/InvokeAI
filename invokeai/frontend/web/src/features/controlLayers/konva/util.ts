import {
  CA_LAYER_NAME,
  INITIAL_IMAGE_LAYER_NAME,
  RASTER_LAYER_BRUSH_LINE_NAME,
  RASTER_LAYER_ERASER_LINE_NAME,
  RASTER_LAYER_IMAGE_NAME,
  RASTER_LAYER_NAME,
  RASTER_LAYER_RECT_SHAPE_NAME,
  RG_LAYER_BRUSH_LINE_NAME,
  RG_LAYER_ERASER_LINE_NAME,
  RG_LAYER_NAME,
  RG_LAYER_RECT_SHAPE_NAME,
} from 'features/controlLayers/konva/naming';
import type Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import type { Vector2d } from 'konva/lib/types';

//#region getScaledFlooredCursorPosition
/**
 * Gets the scaled and floored cursor position on the stage. If the cursor is not currently over the stage, returns null.
 * @param stage The konva stage
 */
export const getScaledFlooredCursorPosition = (stage: Konva.Stage): Vector2d | null => {
  const pointerPosition = stage.getPointerPosition();
  const stageTransform = stage.getAbsoluteTransform().copy();
  if (!pointerPosition) {
    return null;
  }
  const scaledCursorPosition = stageTransform.invert().point(pointerPosition);
  return {
    x: Math.floor(scaledCursorPosition.x),
    y: Math.floor(scaledCursorPosition.y),
  };
};
//#endregion

//#region snapPosToStage
/**
 * Snaps a position to the edge of the stage if within a threshold of the edge
 * @param pos The position to snap
 * @param stage The konva stage
 * @param snapPx The snap threshold in pixels
 */
export const snapPosToStage = (pos: Vector2d, stage: Konva.Stage, snapPx = 10): Vector2d => {
  const snappedPos = { ...pos };
  // Get the normalized threshold for snapping to the edge of the stage
  const thresholdX = snapPx / stage.scaleX();
  const thresholdY = snapPx / stage.scaleY();
  const stageWidth = stage.width() / stage.scaleX();
  const stageHeight = stage.height() / stage.scaleY();
  // Snap to the edge of the stage if within threshold
  if (pos.x - thresholdX < 0) {
    snappedPos.x = 0;
  } else if (pos.x + thresholdX > stageWidth) {
    snappedPos.x = Math.floor(stageWidth);
  }
  if (pos.y - thresholdY < 0) {
    snappedPos.y = 0;
  } else if (pos.y + thresholdY > stageHeight) {
    snappedPos.y = Math.floor(stageHeight);
  }
  return snappedPos;
};
//#endregion

//#region getIsMouseDown
/**
 * Checks if the left mouse button is currently pressed
 * @param e The konva event
 */
export const getIsMouseDown = (e: KonvaEventObject<MouseEvent>): boolean => e.evt.buttons === 1;
//#endregion

//#region getIsFocused
/**
 * Checks if the stage is currently focused
 * @param stage The konva stage
 */
export const getIsFocused = (stage: Konva.Stage): boolean => stage.container().contains(document.activeElement);
//#endregion

//#region mapId
/**
 * Simple util to map an object to its id property. Serves as a minor optimization to avoid recreating a map callback
 * every time we need to map an object to its id, which happens very often.
 * @param object The object with an `id` property
 * @returns The object's id property
 */
export const mapId = (object: { id: string }): string => object.id;
//#endregion

//#region konva selector callbacks
/**
 * Konva selection callback to select all renderable layers. This includes RG, CA II and Raster layers.
 * This can be provided to the `find` or `findOne` konva node methods.
 */
export const selectRenderableLayers = (node: Konva.Node): boolean =>
  node.name() === RG_LAYER_NAME ||
  node.name() === CA_LAYER_NAME ||
  node.name() === INITIAL_IMAGE_LAYER_NAME ||
  node.name() === RASTER_LAYER_NAME;

/**
 * Konva selection callback to select RG mask objects. This includes lines and rects.
 * This can be provided to the `find` or `findOne` konva node methods.
 */
export const selectVectorMaskObjects = (node: Konva.Node): boolean =>
  node.name() === RG_LAYER_BRUSH_LINE_NAME ||
  node.name() === RG_LAYER_ERASER_LINE_NAME ||
  node.name() === RG_LAYER_RECT_SHAPE_NAME;

/**
 * Konva selection callback to select raster layer objects. This includes lines and rects.
 * This can be provided to the `find` or `findOne` konva node methods.
 */
export const selectRasterObjects = (node: Konva.Node): boolean =>
  node.name() === RASTER_LAYER_BRUSH_LINE_NAME ||
  node.name() === RASTER_LAYER_ERASER_LINE_NAME ||
  node.name() === RASTER_LAYER_RECT_SHAPE_NAME ||
  node.name() === RASTER_LAYER_IMAGE_NAME;
//#endregion
