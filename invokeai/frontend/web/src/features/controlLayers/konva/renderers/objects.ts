import { rgbaColorToString } from 'common/util/colorCodeTransformers';
import {
  getLayerBboxId,
  getObjectGroupId,
  IMAGE_PLACEHOLDER_NAME,
  LAYER_BBOX_NAME,
  PREVIEW_GENERATION_BBOX_DUMMY_RECT,
} from 'features/controlLayers/konva/naming';
import type {
  BrushLineObjectRecord,
  EraserLineObjectRecord,
  ImageObjectRecord,
  KonvaEntityAdapter,
  RectShapeObjectRecord,
} from 'features/controlLayers/konva/nodeManager';
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
export const getBrushLine = (
  adapter: KonvaEntityAdapter,
  brushLine: BrushLine,
  name: string
): BrushLineObjectRecord => {
  const objectRecord = adapter.get<BrushLineObjectRecord>(brushLine.id);
  if (objectRecord) {
    return objectRecord;
  }
  const { id, strokeWidth, clip, color } = brushLine;
  const konvaLineGroup = new Konva.Group({
    clip,
    listening: false,
  });
  const konvaLine = new Konva.Line({
    id,
    name,
    strokeWidth,
    tension: 0,
    lineCap: 'round',
    lineJoin: 'round',
    shadowForStrokeEnabled: false,
    globalCompositeOperation: 'source-over',
    listening: false,
    stroke: rgbaColorToString(color),
  });
  return adapter.add({ id, type: 'brush_line', konvaLine, konvaLineGroup });
};

/**
 * Creates a konva line for a eraser line.
 * @param eraserLine The eraser line state
 * @param layerObjectGroup The konva layer's object group to add the line to
 * @param name The konva name for the line
 */
export const getEraserLine = (
  adapter: KonvaEntityAdapter,
  eraserLine: EraserLine,
  name: string
): EraserLineObjectRecord => {
  const objectRecord = adapter.get<EraserLineObjectRecord>(eraserLine.id);
  if (objectRecord) {
    return objectRecord;
  }

  const { id, strokeWidth, clip } = eraserLine;
  const konvaLineGroup = new Konva.Group({
    clip,
    listening: false,
  });
  const konvaLine = new Konva.Line({
    id,
    name,
    strokeWidth,
    tension: 0,
    lineCap: 'round',
    lineJoin: 'round',
    shadowForStrokeEnabled: false,
    globalCompositeOperation: 'destination-out',
    listening: false,
    stroke: rgbaColorToString(DEFAULT_RGBA_COLOR),
  });
  return adapter.add({ id, type: 'eraser_line', konvaLine, konvaLineGroup });
};

/**
 * Creates a konva rect for a rect shape.
 * @param rectShape The rect shape state
 * @param layerObjectGroup The konva layer's object group to add the rect to
 * @param name The konva name for the rect
 */
export const getRectShape = (
  adapter: KonvaEntityAdapter,
  rectShape: RectShape,
  name: string
): RectShapeObjectRecord => {
  const objectRecord = adapter.get<RectShapeObjectRecord>(rectShape.id);
  if (objectRecord) {
    return objectRecord;
  }
  const { id, x, y, width, height } = rectShape;
  const konvaRect = new Konva.Rect({
    id,
    name,
    x,
    y,
    width,
    height,
    listening: false,
    fill: rgbaColorToString(rectShape.color),
  });
  return adapter.add({ id: rectShape.id, type: 'rect_shape', konvaRect });
};

export const updateImageSource = async (arg: {
  objectRecord: ImageObjectRecord;
  image: ImageWithDims;
  getImageDTO?: (imageName: string) => Promise<ImageDTO | null>;
  onLoading?: () => void;
  onLoad?: (konvaImage: Konva.Image) => void;
  onError?: () => void;
}) => {
  const { objectRecord, image, getImageDTO = defaultGetImageDTO, onLoading, onLoad, onError } = arg;

  try {
    objectRecord.isLoading = true;
    if (!objectRecord.konvaImage) {
      objectRecord.konvaPlaceholderGroup.visible(true);
      objectRecord.konvaPlaceholderText.text(t('common.loadingImage', 'Loading Image'));
    }
    onLoading?.();

    const imageDTO = await getImageDTO(image.name);
    if (!imageDTO) {
      objectRecord.imageName = null;
      objectRecord.isLoading = false;
      objectRecord.isError = true;
      objectRecord.konvaPlaceholderGroup.visible(true);
      objectRecord.konvaPlaceholderText.text(t('common.imageFailedToLoad', 'Image Failed to Load'));
      onError?.();
      return;
    }
    const imageEl = new Image();
    imageEl.onload = () => {
      if (objectRecord.konvaImage) {
        objectRecord.konvaImage.setAttrs({
          image: imageEl,
        });
      } else {
        objectRecord.konvaImage = new Konva.Image({
          id: objectRecord.id,
          listening: false,
          image: imageEl,
        });
        objectRecord.konvaImageGroup.add(objectRecord.konvaImage);
        objectRecord.imageName = image.name;
      }
      objectRecord.isLoading = false;
      objectRecord.isError = false;
      objectRecord.konvaPlaceholderGroup.visible(false);
      onLoad?.(objectRecord.konvaImage);
    };
    imageEl.onerror = () => {
      objectRecord.imageName = null;
      objectRecord.isLoading = false;
      objectRecord.isError = true;
      objectRecord.konvaPlaceholderGroup.visible(true);
      objectRecord.konvaPlaceholderText.text(t('common.imageFailedToLoad', 'Image Failed to Load'));
      onError?.();
    };
    imageEl.id = image.name;
    imageEl.src = imageDTO.image_url;
  } catch {
    objectRecord.imageName = null;
    objectRecord.isLoading = false;
    objectRecord.isError = true;
    objectRecord.konvaPlaceholderGroup.visible(true);
    objectRecord.konvaPlaceholderText.text(t('common.imageFailedToLoad', 'Image Failed to Load'));
    onError?.();
  }
};

/**
 * Creates an image placeholder group for an image object.
 * @param image The image object state
 * @returns The konva group for the image placeholder, and callbacks to handle loading and error states
 */
export const createImageObjectGroup = (arg: {
  adapter: KonvaEntityAdapter;
  obj: ImageObject;
  name: string;
  getImageDTO?: (imageName: string) => Promise<ImageDTO | null>;
  onLoad?: (konvaImage: Konva.Image) => void;
  onLoading?: () => void;
  onError?: () => void;
}): ImageObjectRecord => {
  const { adapter, obj, name, getImageDTO = defaultGetImageDTO, onLoad, onLoading, onError } = arg;
  let objectRecord = adapter.get<ImageObjectRecord>(obj.id);
  if (objectRecord) {
    return objectRecord;
  }
  const { id, image } = obj;
  const { width, height } = obj;
  const konvaImageGroup = new Konva.Group({ id, name, listening: false, x: obj.x, y: obj.y });
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
  objectRecord = adapter.add({
    id,
    type: 'image',
    konvaImageGroup,
    konvaPlaceholderGroup,
    konvaPlaceholderRect,
    konvaPlaceholderText,
    konvaImage: null,
    imageName: null,
    isLoading: false,
    isError: false,
  });
  updateImageSource({ objectRecord, image, getImageDTO, onLoad, onLoading, onError });
  return objectRecord;
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
