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
  IMAGE_PLACEHOLDER_NAME,
  LAYER_BBOX_NAME,
  PREVIEW_GENERATION_BBOX_DUMMY_RECT,
} from 'features/controlLayers/konva/naming';
import type {
  BrushLine,
  CanvasEntity,
  EraserLine,
  ImageObject,
  ImageWithDims,
  RectShape,
} from 'features/controlLayers/store/types';
import { DEFAULT_RGBA_COLOR } from 'features/controlLayers/store/types';
import { t } from 'i18next';
import Konva from 'konva';
import { getImageDTO as defaultGetImageDTO } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';
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

export const updateImageSource = async (arg: {
  entry: ImageEntry;
  image: ImageWithDims;
  getImageDTO?: (imageName: string) => Promise<ImageDTO | null>;
  onLoading?: () => void;
  onLoad?: (konvaImage: Konva.Image) => void;
  onError?: () => void;
}) => {
  const { entry, image, getImageDTO = defaultGetImageDTO, onLoading, onLoad, onError } = arg;

  try {
    entry.isLoading = true;
    if (!entry.konvaImage) {
      entry.konvaPlaceholderGroup.visible(true);
      entry.konvaPlaceholderText.text(t('common.loadingImage', 'Loading Image'));
    }
    onLoading?.();

    const imageDTO = await getImageDTO(image.name);
    if (!imageDTO) {
      entry.isLoading = false;
      entry.isError = true;
      entry.konvaPlaceholderGroup.visible(true);
      entry.konvaPlaceholderText.text(t('common.imageFailedToLoad', 'Image Failed to Load'));
      onError?.();
      return;
    }
    const imageEl = new Image();
    imageEl.onload = () => {
      if (entry.konvaImage) {
        entry.konvaImage.setAttrs({
          image: imageEl,
        });
      } else {
        entry.konvaImage = new Konva.Image({
          id: entry.id,
          listening: false,
          image: imageEl,
        });
        entry.konvaImageGroup.add(entry.konvaImage);
      }
      entry.isLoading = false;
      entry.isError = false;
      entry.konvaPlaceholderGroup.visible(false);
      onLoad?.(entry.konvaImage);
    };
    imageEl.onerror = () => {
      entry.isLoading = false;
      entry.isError = true;
      entry.konvaPlaceholderGroup.visible(true);
      entry.konvaPlaceholderText.text(t('common.imageFailedToLoad', 'Image Failed to Load'));
      onError?.();
    };
    imageEl.id = image.name;
    imageEl.src = imageDTO.image_url;
  } catch {
    entry.isLoading = false;
    entry.isError = true;
    entry.konvaPlaceholderGroup.visible(true);
    entry.konvaPlaceholderText.text(t('common.imageFailedToLoad', 'Image Failed to Load'));
    onError?.();
  }
};

/**
 * Creates an image placeholder group for an image object.
 * @param image The image object state
 * @returns The konva group for the image placeholder, and callbacks to handle loading and error states
 */
export const createImageObjectGroup = (arg: {
  mapping: EntityToKonvaMapping;
  obj: ImageObject;
  name: string;
  getImageDTO?: (imageName: string) => Promise<ImageDTO | null>;
  onLoad?: (konvaImage: Konva.Image) => void;
  onLoading?: () => void;
  onError?: () => void;
}): ImageEntry => {
  const { mapping, obj, name, getImageDTO = defaultGetImageDTO, onLoad, onLoading, onError } = arg;
  let entry = mapping.getEntry<ImageEntry>(obj.id);
  if (entry) {
    return entry;
  }
  const { id, image } = obj;
  const { width, height } = obj;
  const konvaImageGroup = new Konva.Group({ id, name, listening: false });
  const konvaPlaceholderGroup = new Konva.Group({ name: IMAGE_PLACEHOLDER_NAME, listening: false });
  const konvaPlaceholderRect = new Konva.Rect({
    fill: 'hsl(220 12% 45% / 1)', // 'base.500'
    width,
    height,
    listening: false,
  });
  const konvaPlaceholderText = new Konva.Text({
    fill: 'hsl(220 12% 10% / 1)', // 'base.900'
    width,
    height,
    align: 'center',
    verticalAlign: 'middle',
    fontFamily: '"Inter Variable", sans-serif',
    fontSize: width / 16,
    fontStyle: '600',
    text: t('common.loadingImage', 'Loading Image'),
    listening: false,
  });
  konvaPlaceholderGroup.add(konvaPlaceholderRect);
  konvaPlaceholderGroup.add(konvaPlaceholderText);
  konvaImageGroup.add(konvaPlaceholderGroup);
  mapping.konvaObjectGroup.add(konvaImageGroup);

  entry = mapping.addEntry({
    id,
    type: 'image',
    konvaImageGroup,
    konvaPlaceholderGroup,
    konvaPlaceholderText,
    konvaImage: null,
    isLoading: false,
    isError: false,
  });
  updateImageSource({ entry, image, getImageDTO, onLoad, onLoading, onError });
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
