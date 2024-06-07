import { rgbaColorToString } from 'features/canvas/util/colorToString';
import {
  getLayerBboxId,
  getObjectGroupId,
  LAYER_BBOX_NAME,
  TOOL_PREVIEW_IMAGE_DIMS_RECT,
} from 'features/controlLayers/konva/naming';
import type { BrushLine, EraserLine, ImageObject, Layer, RectShape } from 'features/controlLayers/store/types';
import { DEFAULT_RGBA_COLOR } from 'features/controlLayers/store/types';
import { t } from 'i18next';
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

/**
 * Creates an image placeholder group for an image object.
 * @param imageObject The image object state
 * @returns The konva group for the image placeholder, and callbacks to handle loading and error states
 */
const createImagePlaceholderGroup = (
  imageObject: ImageObject
): { konvaPlaceholderGroup: Konva.Group; onError: () => void; onLoading: () => void; onLoaded: () => void } => {
  const { width, height } = imageObject.image;
  const konvaPlaceholderGroup = new Konva.Group({ name: 'image-placeholder', listening: false });
  const konvaPlaceholderRect = new Konva.Rect({
    fill: 'hsl(220 12% 45% / 1)', // 'base.500'
    width,
    height,
  });
  const konvaPlaceholderText = new Konva.Text({
    name: 'image-placeholder-text',
    fill: 'hsl(220 12% 10% / 1)', // 'base.900'
    width,
    height,
    align: 'center',
    verticalAlign: 'middle',
    fontFamily: '"Inter Variable", sans-serif',
    fontSize: width / 16,
    fontStyle: '600',
    text: 'Loading Image',
    listening: false,
  });
  konvaPlaceholderGroup.add(konvaPlaceholderRect);
  konvaPlaceholderGroup.add(konvaPlaceholderText);

  const onError = () => {
    konvaPlaceholderText.text(t('common.imageFailedToLoad', 'Image Failed to Load'));
  };
  const onLoading = () => {
    konvaPlaceholderText.text(t('common.loadingImage', 'Loading Image'));
  };
  const onLoaded = () => {
    konvaPlaceholderGroup.destroy();
  };
  return { konvaPlaceholderGroup, onError, onLoading, onLoaded };
};

/**
 * Creates an image object group. Because images are loaded asynchronously, and we need to handle loading an error state,
 * the image is rendered in a group, which includes a placeholder.
 * @param imageObject The image object state
 * @param layerObjectGroup The konva layer's object group to add the image to
 * @param name The konva name for the image
 * @returns A promise that resolves to the konva group for the image object
 */
export const createImageObjectGroup = async (
  imageObject: ImageObject,
  layerObjectGroup: Konva.Group,
  name: string
): Promise<Konva.Group> => {
  const konvaImageGroup = new Konva.Group({ id: imageObject.id, name, listening: false });
  const placeholder = createImagePlaceholderGroup(imageObject);
  konvaImageGroup.add(placeholder.konvaPlaceholderGroup);
  layerObjectGroup.add(konvaImageGroup);
  getImageDTO(imageObject.image.name).then((imageDTO) => {
    if (!imageDTO) {
      placeholder.onError();
      return;
    }
    const imageEl = new Image();
    imageEl.onload = () => {
      const konvaImage = new Konva.Image({
        id: imageObject.id,
        name,
        listening: false,
        image: imageEl,
      });
      placeholder.onLoaded();
      konvaImageGroup.add(konvaImage);
    };
    imageEl.onerror = () => {
      placeholder.onError();
    };
    imageEl.id = imageObject.id;
    imageEl.src = imageDTO.image_url;
  });
  return konvaImageGroup;
};

/**
 * Creates a bounding box rect for a layer.
 * @param layerState The layer state for the layer to create the bounding box for
 * @param konvaLayer The konva layer to attach the bounding box to
 */
export const createBboxRect = (layerState: Layer, konvaLayer: Konva.Layer): Konva.Rect => {
  const rect = new Konva.Rect({
    id: getLayerBboxId(layerState.id),
    name: LAYER_BBOX_NAME,
    strokeWidth: 1,
    visible: false,
  });
  konvaLayer.add(rect);
  return rect;
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

export const createImageDimsPreview = (konvaLayer: Konva.Layer, width: number, height: number): Konva.Rect => {
  const imageDimsPreview = new Konva.Rect({
    id: TOOL_PREVIEW_IMAGE_DIMS_RECT,
    x: 0,
    y: 0,
    width,
    height,
    stroke: 'rgb(255,0,255)',
    strokeWidth: 1 / konvaLayer.getStage().scaleX(),
    listening: false,
  });
  konvaLayer.add(imageDimsPreview);
  return imageDimsPreview;
};
