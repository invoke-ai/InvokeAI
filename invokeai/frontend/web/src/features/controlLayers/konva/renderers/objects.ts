import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import type {
  BrushLineEntry,
  EntityToKonvaMapping,
  EraserLineEntry,
  ImageEntry,
  RectShapeEntry,
} from 'features/controlLayers/konva/entityToKonvaMap';
import {
  getLayerBboxId,
  getObjectGroupId,
  LAYER_BBOX_NAME,
  PREVIEW_GENERATION_BBOX_DUMMY_RECT,
} from 'features/controlLayers/konva/naming';
import type { BrushLine, CanvasEntity, EraserLine, ImageObject, RectShape } from 'features/controlLayers/store/types';
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
export const getBrushLine = (mapping: EntityToKonvaMapping, brushLine: BrushLine, name: string): BrushLineEntry => {
  let entry = mapping.getEntry<BrushLineEntry>(brushLine.id);
  if (entry) {
    return entry;
  }

  const konvaLineGroup = new Konva.Group({
    clip: brushLine.clip,
    listening: false,
  });
  const konvaLine = new Konva.Line({
    id: brushLine.id,
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
  konvaLineGroup.add(konvaLine);
  mapping.konvaObjectGroup.add(konvaLineGroup);
  entry = mapping.addEntry({ id: brushLine.id, type: 'brush_line', konvaLine, konvaLineGroup });
  return entry;
};

/**
 * Creates a konva line for a eraser line.
 * @param eraserLine The eraser line state
 * @param layerObjectGroup The konva layer's object group to add the line to
 * @param name The konva name for the line
 */
export const getEraserLine = (mapping: EntityToKonvaMapping, eraserLine: EraserLine, name: string): EraserLineEntry => {
  let entry = mapping.getEntry<EraserLineEntry>(eraserLine.id);
  if (entry) {
    return entry;
  }

  const konvaLineGroup = new Konva.Group({
    clip: eraserLine.clip,
    listening: false,
  });
  const konvaLine = new Konva.Line({
    id: eraserLine.id,
    name,
    strokeWidth: eraserLine.strokeWidth,
    tension: 0,
    lineCap: 'round',
    lineJoin: 'round',
    shadowForStrokeEnabled: false,
    globalCompositeOperation: 'destination-out',
    listening: false,
    stroke: rgbaColorToString(DEFAULT_RGBA_COLOR),
    clip: eraserLine.clip,
  });
  konvaLineGroup.add(konvaLine);
  mapping.konvaObjectGroup.add(konvaLineGroup);
  entry = mapping.addEntry({ id: eraserLine.id, type: 'eraser_line', konvaLine, konvaLineGroup });
  return entry;
};

/**
 * Creates a konva rect for a rect shape.
 * @param rectShape The rect shape state
 * @param layerObjectGroup The konva layer's object group to add the rect to
 * @param name The konva name for the rect
 */
export const getRectShape = (mapping: EntityToKonvaMapping, rectShape: RectShape, name: string): RectShapeEntry => {
  let entry = mapping.getEntry<RectShapeEntry>(rectShape.id);
  if (entry) {
    return entry;
  }
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
  mapping.konvaObjectGroup.add(konvaRect);
  entry = mapping.addEntry({ id: rectShape.id, type: 'rect_shape', konvaRect });
  return entry;
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
    listening: false,
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
  mapping: EntityToKonvaMapping,
  imageObject: ImageObject,
  name: string
): Promise<ImageEntry> => {
  let entry = mapping.getEntry<ImageEntry>(imageObject.id);
  if (entry) {
    return entry;
  }
  const konvaImageGroup = new Konva.Group({ id: imageObject.id, name, listening: false });
  const placeholder = createImagePlaceholderGroup(imageObject);
  konvaImageGroup.add(placeholder.konvaPlaceholderGroup);
  mapping.konvaObjectGroup.add(konvaImageGroup);

  entry = mapping.addEntry({ id: imageObject.id, type: 'image', konvaGroup: konvaImageGroup, konvaImage: null });
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
      entry.konvaImage = konvaImage;
    };
    imageEl.onerror = () => {
      placeholder.onError();
    };
    imageEl.id = imageObject.id;
    imageEl.src = imageDTO.image_url;
  });
  return entry;
};

/**
 * Creates a bounding box rect for a layer.
 * @param entity The layer state for the layer to create the bounding box for
 * @param konvaLayer The konva layer to attach the bounding box to
 */
export const createBboxRect = (entity: CanvasEntity, konvaLayer: Konva.Layer): Konva.Rect => {
  const rect = new Konva.Rect({
    id: getLayerBboxId(entity.id),
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
    id: PREVIEW_GENERATION_BBOX_DUMMY_RECT,
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
