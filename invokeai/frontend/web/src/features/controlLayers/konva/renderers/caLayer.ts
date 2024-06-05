import { LightnessToAlphaFilter } from 'features/controlLayers/konva/filters';
import { CA_LAYER_IMAGE_NAME, CA_LAYER_NAME, getCALayerImageId } from 'features/controlLayers/konva/naming';
import type { ControlAdapterLayer } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { ImageDTO } from 'services/api/types';

/**
 * Logic for creating and rendering control adapter (control net & t2i adapter) layers. These layers have image objects
 * and require some special handling to update the source and attributes as control images are swapped or processed.
 */

/**
 * Creates a control adapter layer.
 * @param stage The konva stage
 * @param layerState The control adapter layer state
 */
const createCALayer = (stage: Konva.Stage, layerState: ControlAdapterLayer): Konva.Layer => {
  const konvaLayer = new Konva.Layer({
    id: layerState.id,
    name: CA_LAYER_NAME,
    imageSmoothingEnabled: true,
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
  });
  konvaLayer.add(konvaImage);
  return konvaImage;
};

/**
 * Updates the image source for a control adapter layer. This includes loading the image from the server and updating
 * the konva image.
 * @param stage The konva stage
 * @param konvaLayer The konva layer
 * @param layerState The control adapter layer state
 * @param getImageDTO A function to retrieve an image DTO from the server, used to update the image source
 */
const updateCALayerImageSource = async (
  stage: Konva.Stage,
  konvaLayer: Konva.Layer,
  layerState: ControlAdapterLayer,
  getImageDTO: (imageName: string) => Promise<ImageDTO | null>
): Promise<void> => {
  const image = layerState.controlAdapter.processedImage ?? layerState.controlAdapter.image;
  if (image) {
    const imageName = image.name;
    const imageDTO = await getImageDTO(imageName);
    if (!imageDTO) {
      return;
    }
    const imageEl = new Image();
    const imageId = getCALayerImageId(layerState.id, imageName);
    imageEl.onload = () => {
      // Find the existing image or create a new one - must find using the name, bc the id may have just changed
      const konvaImage =
        konvaLayer.findOne<Konva.Image>(`.${CA_LAYER_IMAGE_NAME}`) ?? createCALayerImage(konvaLayer, imageEl);

      // Update the image's attributes
      konvaImage.setAttrs({
        id: imageId,
        image: imageEl,
      });
      updateCALayerImageAttrs(stage, konvaImage, layerState);
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
 * @param layerState The control adapter layer state
 */

const updateCALayerImageAttrs = (
  stage: Konva.Stage,
  konvaImage: Konva.Image,
  layerState: ControlAdapterLayer
): void => {
  let needsCache = false;
  // Konva erroneously reports NaN for width and height when the stage is hidden. This causes errors when caching,
  // but it doesn't seem to break anything.
  // TODO(psyche): Investigate and report upstream.
  const newWidth = stage.width() / stage.scaleX();
  const newHeight = stage.height() / stage.scaleY();
  const hasFilter = konvaImage.filters() !== null && konvaImage.filters().length > 0;
  if (
    konvaImage.width() !== newWidth ||
    konvaImage.height() !== newHeight ||
    konvaImage.visible() !== layerState.isEnabled ||
    hasFilter !== layerState.isFilterEnabled
  ) {
    konvaImage.setAttrs({
      opacity: layerState.opacity,
      scaleX: 1,
      scaleY: 1,
      width: stage.width() / stage.scaleX(),
      height: stage.height() / stage.scaleY(),
      visible: layerState.isEnabled,
      filters: layerState.isFilterEnabled ? [LightnessToAlphaFilter] : [],
    });
    needsCache = true;
  }
  if (konvaImage.opacity() !== layerState.opacity) {
    konvaImage.opacity(layerState.opacity);
  }
  if (needsCache) {
    konvaImage.cache();
  }
};

/**
 * Renders a control adapter layer. If the layer doesn't already exist, it is created. Otherwise, the layer is updated
 * with the current image source and attributes.
 * @param stage The konva stage
 * @param layerState The control adapter layer state
 * @param getImageDTO A function to retrieve an image DTO from the server, used to update the image source
 */
export const renderCALayer = (
  stage: Konva.Stage,
  layerState: ControlAdapterLayer,
  getImageDTO: (imageName: string) => Promise<ImageDTO | null>
): void => {
  const konvaLayer = stage.findOne<Konva.Layer>(`#${layerState.id}`) ?? createCALayer(stage, layerState);
  const konvaImage = konvaLayer.findOne<Konva.Image>(`.${CA_LAYER_IMAGE_NAME}`);
  const canvasImageSource = konvaImage?.image();
  let imageSourceNeedsUpdate = false;
  if (canvasImageSource instanceof HTMLImageElement) {
    const image = layerState.controlAdapter.processedImage ?? layerState.controlAdapter.image;
    if (image && canvasImageSource.id !== getCALayerImageId(layerState.id, image.name)) {
      imageSourceNeedsUpdate = true;
    } else if (!image) {
      imageSourceNeedsUpdate = true;
    }
  } else if (!canvasImageSource) {
    imageSourceNeedsUpdate = true;
  }

  if (imageSourceNeedsUpdate) {
    updateCALayerImageSource(stage, konvaLayer, layerState, getImageDTO);
  } else if (konvaImage) {
    updateCALayerImageAttrs(stage, konvaImage, layerState);
  }
};
