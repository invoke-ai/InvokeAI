import { LightnessToAlphaFilter } from 'features/controlLayers/konva/filters';
import type { EntityToKonvaMap } from 'features/controlLayers/konva/konvaMap';
import { CA_LAYER_IMAGE_NAME, CA_LAYER_NAME, getCAImageId } from 'features/controlLayers/konva/naming';
import type { ControlAdapterEntity } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { ImageDTO } from 'services/api/types';

/**
 * Logic for creating and rendering control adapter (control net & t2i adapter) layers. These layers have image objects
 * and require some special handling to update the source and attributes as control images are swapped or processed.
 */

/**
 * Creates a control adapter layer.
 * @param stage The konva stage
 * @param ca The control adapter layer state
 */
const createCALayer = (stage: Konva.Stage, ca: ControlAdapterEntity): Konva.Layer => {
  const konvaLayer = new Konva.Layer({
    id: ca.id,
    name: CA_LAYER_NAME,
    imageSmoothingEnabled: false,
    listening: false,
  });
  stage.add(konvaLayer);
  return konvaLayer;
};

/**
 * Creates a control adapter layer image.
 * @param konvaLayer The konva layer
 * @param imageEl The image element
 */
const createCALayerImage = (konvaLayer: Konva.Layer, imageEl: HTMLImageElement): Konva.Image => {
  const konvaImage = new Konva.Image({
    name: CA_LAYER_IMAGE_NAME,
    image: imageEl,
    listening: false,
  });
  konvaLayer.add(konvaImage);
  return konvaImage;
};

/**
 * Updates the image source for a control adapter layer. This includes loading the image from the server and updating
 * the konva image.
 * @param stage The konva stage
 * @param konvaLayer The konva layer
 * @param ca The control adapter layer state
 * @param getImageDTO A function to retrieve an image DTO from the server, used to update the image source
 */
const updateCALayerImageSource = async (
  stage: Konva.Stage,
  konvaLayer: Konva.Layer,
  ca: ControlAdapterEntity,
  getImageDTO: (imageName: string) => Promise<ImageDTO | null>
): Promise<void> => {
  const image = ca.processedImage ?? ca.image;
  if (image) {
    const imageName = image.name;
    const imageDTO = await getImageDTO(imageName);
    if (!imageDTO) {
      return;
    }
    const imageEl = new Image();
    const imageId = getCAImageId(ca.id, imageName);
    imageEl.onload = () => {
      // Find the existing image or create a new one - must find using the name, bc the id may have just changed
      const konvaImage =
        konvaLayer.findOne<Konva.Image>(`.${CA_LAYER_IMAGE_NAME}`) ?? createCALayerImage(konvaLayer, imageEl);

      // Update the image's attributes
      konvaImage.setAttrs({
        id: imageId,
        image: imageEl,
      });
      updateCALayerImageAttrs(stage, konvaImage, ca);
      // Must cache after this to apply the filters
      konvaImage.cache();
      imageEl.id = imageId;
    };
    imageEl.src = imageDTO.image_url;
  } else {
    konvaLayer.findOne(`.${CA_LAYER_IMAGE_NAME}`)?.destroy();
  }
};

/**
 * Updates the image attributes for a control adapter layer's image (width, height, visibility, opacity, filters).
 * @param stage The konva stage
 * @param konvaImage The konva image
 * @param ca The control adapter layer state
 */

const updateCALayerImageAttrs = (stage: Konva.Stage, konvaImage: Konva.Image, ca: ControlAdapterEntity): void => {
  let needsCache = false;
  // TODO(psyche): `node.filters()` returns null if no filters; report upstream
  const filters = konvaImage.filters() ?? [];
  const filter = filters[0] ?? null;
  const filterNeedsUpdate = (filter === null && ca.filter !== 'none') || (filter && filter.name !== ca.filter);
  if (
    konvaImage.x() !== ca.x ||
    konvaImage.y() !== ca.y ||
    konvaImage.visible() !== ca.isEnabled ||
    filterNeedsUpdate
  ) {
    konvaImage.setAttrs({
      opacity: ca.opacity,
      scaleX: 1,
      scaleY: 1,
      visible: ca.isEnabled,
      filters: ca.filter === 'LightnessToAlphaFilter' ? [LightnessToAlphaFilter] : [],
    });
    needsCache = true;
  }
  if (konvaImage.opacity() !== ca.opacity) {
    konvaImage.opacity(ca.opacity);
  }
  if (needsCache) {
    konvaImage.cache();
  }
};

/**
 * Renders a control adapter layer. If the layer doesn't already exist, it is created. Otherwise, the layer is updated
 * with the current image source and attributes.
 * @param stage The konva stage
 * @param ca The control adapter layer state
 * @param getImageDTO A function to retrieve an image DTO from the server, used to update the image source
 */
export const renderCALayer = (
  stage: Konva.Stage,
  controlAdapterMap: EntityToKonvaMap,
  ca: ControlAdapterEntity,
  getImageDTO: (imageName: string) => Promise<ImageDTO | null>
): void => {
  const konvaLayer = stage.findOne<Konva.Layer>(`#${ca.id}`) ?? createCALayer(stage, ca);
  const konvaImage = konvaLayer.findOne<Konva.Image>(`.${CA_LAYER_IMAGE_NAME}`);
  const canvasImageSource = konvaImage?.image();

  let imageSourceNeedsUpdate = false;

  if (canvasImageSource instanceof HTMLImageElement) {
    const image = ca.processedImage ?? ca.image;
    if (image && canvasImageSource.id !== getCAImageId(ca.id, image.name)) {
      imageSourceNeedsUpdate = true;
    } else if (!image) {
      imageSourceNeedsUpdate = true;
    }
  } else if (!canvasImageSource) {
    imageSourceNeedsUpdate = true;
  }

  if (imageSourceNeedsUpdate) {
    updateCALayerImageSource(stage, konvaLayer, ca, getImageDTO);
  } else if (konvaImage) {
    updateCALayerImageAttrs(stage, konvaImage, ca);
  }
};

export const renderControlAdapters = (
  stage: Konva.Stage,
  controlAdapterMap: EntityToKonvaMap,
  controlAdapters: ControlAdapterEntity[],
  getImageDTO: (imageName: string) => Promise<ImageDTO | null>
): void => {
  // Destroy nonexistent layers
  for (const mapping of controlAdapterMap.getMappings()) {
    if (!controlAdapters.find((ca) => ca.id === mapping.id)) {
      controlAdapterMap.destroyMapping(mapping.id);
    }
  }
  for (const ca of controlAdapters) {
    renderCALayer(stage, controlAdapterMap, ca, getImageDTO);
  }
};
