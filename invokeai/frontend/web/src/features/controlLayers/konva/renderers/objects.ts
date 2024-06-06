import { rgbaColorToString } from 'features/canvas/util/colorToString';
import { getObjectGroupId } from 'features/controlLayers/konva/naming';
import type { BrushLine, EraserLine, ImageObject, RectShape } from 'features/controlLayers/store/types';
import { DEFAULT_RGBA_COLOR } from 'features/controlLayers/store/types';
import Konva from 'konva';
import { getImageDTO } from 'services/api/endpoints/images';
import { v4 as uuidv4 } from 'uuid';

/**
 * Utilities to create various konva objects from layer state. These are used by both the raster and regional guidance
 * layers types.
 */

/**
 * Creates a konva line for a brush line.
 * @param brushLine The brush line state
 * @param layerObjectGroup The konva layer's object group to add the line to
 * @param name The konva name for the line
 */
export const createBrushLine = (brushLine: BrushLine, layerObjectGroup: Konva.Group, name: string): Konva.Line => {
  const konvaLine = new Konva.Line({
    id: brushLine.id,
    key: brushLine.id,
    name,
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
 * @param name The konva name for the line
 */
export const createEraserLine = (eraserLine: EraserLine, layerObjectGroup: Konva.Group, name: string): Konva.Line => {
  const konvaLine = new Konva.Line({
    id: eraserLine.id,
    key: eraserLine.id,
    name,
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
 * @param layerObjectGroup The konva layer's object group to add the rect to
 * @param name The konva name for the rect
 */
export const createRectShape = (rectShape: RectShape, layerObjectGroup: Konva.Group, name: string): Konva.Rect => {
  const konvaRect = new Konva.Rect({
    id: rectShape.id,
    key: rectShape.id,
    name,
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

export const createImageObject = async (
  imageObject: ImageObject,
  layerObjectGroup: Konva.Group,
  name: string
): Promise<Konva.Image | null> => {
  const imageDTO = await getImageDTO(imageObject.image.name);
  if (!imageDTO) {
    return null;
  }
  return new Promise((resolve) => {
    const imageEl = new Image();
    imageEl.onload = () => {
      const konvaImage = new Konva.Image({
        id: imageObject.id,
        name,
        listening: false,
        image: imageEl,
      });
      layerObjectGroup.add(konvaImage);
      resolve(konvaImage);
    };
    imageEl.onerror = () => {
      resolve(null);
    };
    imageEl.id = imageObject.id;
    imageEl.src = imageDTO.image_url;
  });
};
/**
 * Creates a konva group for a layer's objects.
 * @param konvaLayer The konva layer to add the object group to
 * @param name The konva name for the group
 * @returns
 */
export const createObjectGroup = (konvaLayer: Konva.Layer, name: string): Konva.Group => {
  const konvaObjectGroup = new Konva.Group({
    id: getObjectGroupId(konvaLayer.id(), uuidv4()),
    name,
    listening: false,
  });
  konvaLayer.add(konvaObjectGroup);
  return konvaObjectGroup;
};
