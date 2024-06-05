import { rgbaColorToString } from 'features/canvas/util/colorToString';
import { RG_LAYER_LINE_NAME, RG_LAYER_RECT_NAME } from 'features/controlLayers/konva/naming';
import type { BrushLine, EraserLine, RectShape } from 'features/controlLayers/store/types';
import { DEFAULT_RGBA_COLOR } from 'features/controlLayers/store/types';
import Konva from 'konva';

/**
 * Utilities to create various konva objects from layer state. These are used by both the raster and regional guidance
 * layers types.
 */

/**
 * Creates a konva line for a brush line.
 * @param brushLine The brush line state
 * @param layerObjectGroup The konva layer's object group to add the line to
 */
export const createBrushLine = (brushLine: BrushLine, layerObjectGroup: Konva.Group): Konva.Line => {
  const konvaLine = new Konva.Line({
    id: brushLine.id,
    key: brushLine.id,
    name: RG_LAYER_LINE_NAME,
    strokeWidth: brushLine.strokeWidth,
    tension: 0,
    lineCap: 'round',
    lineJoin: 'round',
    shadowForStrokeEnabled: false,
    globalCompositeOperation: 'source-over',
    listening: false,
    stroke: rgbaColorToString(brushLine.color),
  });
  layerObjectGroup.add(konvaLine);
  return konvaLine;
};

/**
 * Creates a konva line for a eraser line.
 * @param eraserLine The eraser line state
 * @param layerObjectGroup The konva layer's object group to add the line to
 */
export const createEraserLine = (eraserLine: EraserLine, layerObjectGroup: Konva.Group): Konva.Line => {
  const konvaLine = new Konva.Line({
    id: eraserLine.id,
    key: eraserLine.id,
    name: RG_LAYER_LINE_NAME,
    strokeWidth: eraserLine.strokeWidth,
    tension: 0,
    lineCap: 'round',
    lineJoin: 'round',
    shadowForStrokeEnabled: false,
    globalCompositeOperation: 'destination-out',
    listening: false,
    stroke: rgbaColorToString(DEFAULT_RGBA_COLOR),
  });
  layerObjectGroup.add(konvaLine);
  return konvaLine;
};

/**
 * Creates a konva rect for a rect shape.
 * @param rectShape The rect shape state
 * @param layerObjectGroup The konva layer's object group to add the line to
 */
export const createRectShape = (rectShape: RectShape, layerObjectGroup: Konva.Group): Konva.Rect => {
  const konvaRect = new Konva.Rect({
    id: rectShape.id,
    key: rectShape.id,
    name: RG_LAYER_RECT_NAME,
    x: rectShape.x,
    y: rectShape.y,
    width: rectShape.width,
    height: rectShape.height,
    listening: false,
    fill: rgbaColorToString(rectShape.color),
  });
  layerObjectGroup.add(konvaRect);
  return konvaRect;
};
