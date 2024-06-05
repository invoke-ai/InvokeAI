import {
  getCALayerImageId,
  getIILayerImageId,
  INITIAL_IMAGE_LAYER_IMAGE_NAME,
  INITIAL_IMAGE_LAYER_NAME,
} from 'features/controlLayers/konva/naming';
import type { InitialImageLayer } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { ImageDTO } from 'services/api/types';

/**
 * Logic for creating and rendering initial image layers. Well, just the one, actually, because it's a singleton.
 * TODO(psyche): Raster layers effectively supersede the initial image layer type.
 */

/**
 * Creates an initial image konva layer.
 * @param stage The konva stage
 * @param layerState The initial image layer state
 */
const createIILayer = (stage: Konva.Stage, layerState: InitialImageLayer): Konva.Layer => {
  const konvaLayer = new Konva.Layer({
    id: layerState.id,
    name: INITIAL_IMAGE_LAYER_NAME,
    imageSmoothingEnabled: true,
    listening: false,
  });
  stage.add(konvaLayer);
  return konvaLayer;
};

/**
 * Creates the konva image for an initial image layer.
 * @param konvaLayer The konva layer
 * @param imageEl The image element
 */
const createIILayerImage = (konvaLayer: Konva.Layer, imageEl: HTMLImageElement): Konva.Image => {
  const konvaImage = new Konva.Image({
    name: INITIAL_IMAGE_LAYER_IMAGE_NAME,
    image: imageEl,
  });
  konvaLayer.add(konvaImage);
  return konvaImage;
};

/**
 * Updates an initial image layer's attributes (width, height, opacity, visibility).
 * @param stage The konva stage
 * @param konvaImage The konva image
 * @param layerState The initial image layer state
 */
const updateIILayerImageAttrs = (stage: Konva.Stage, konvaImage: Konva.Image, layerState: InitialImageLayer): void => {
  // Konva erroneously reports NaN for width and height when the stage is hidden. This causes errors when caching,
  // but it doesn't seem to break anything.
  // TODO(psyche): Investigate and report upstream.
  const newWidth = stage.width() / stage.scaleX();
  const newHeight = stage.height() / stage.scaleY();
  if (
    konvaImage.width() !== newWidth ||
    konvaImage.height() !== newHeight ||
    konvaImage.visible() !== layerState.isEnabled
  ) {
    konvaImage.setAttrs({
      opacity: layerState.opacity,
      scaleX: 1,
      scaleY: 1,
      width: stage.width() / stage.scaleX(),
      height: stage.height() / stage.scaleY(),
      visible: layerState.isEnabled,
    });
  }
  if (konvaImage.opacity() !== layerState.opacity) {
    konvaImage.opacity(layerState.opacity);
  }
};

/**
 * Update an initial image layer's image source when the image changes.
 * @param stage The konva stage
 * @param konvaLayer The konva layer
 * @param layerState The initial image layer state
 * @param getImageDTO A function to retrieve an image DTO from the server, used to update the image source
 */
const updateIILayerImageSource = async (
  stage: Konva.Stage,
  konvaLayer: Konva.Layer,
  layerState: InitialImageLayer,
  getImageDTO: (imageName: string) => Promise<ImageDTO | null>
): Promise<void> => {
  if (layerState.image) {
    const imageName = layerState.image.name;
    const imageDTO = await getImageDTO(imageName);
    if (!imageDTO) {
      return;
    }
    const imageEl = new Image();
    const imageId = getIILayerImageId(layerState.id, imageName);
    imageEl.onload = () => {
      // Find the existing image or create a new one - must find using the name, bc the id may have just changed
      const konvaImage =
        konvaLayer.findOne<Konva.Image>(`.${INITIAL_IMAGE_LAYER_IMAGE_NAME}`) ??
        createIILayerImage(konvaLayer, imageEl);

      // Update the image's attributes
      konvaImage.setAttrs({
        id: imageId,
        image: imageEl,
      });
      updateIILayerImageAttrs(stage, konvaImage, layerState);
      imageEl.id = imageId;
    };
    imageEl.src = imageDTO.image_url;
  } else {
    konvaLayer.findOne(`.${INITIAL_IMAGE_LAYER_IMAGE_NAME}`)?.destroy();
  }
};

/**
 * Renders an initial image layer.
 * @param stage The konva stage
 * @param layerState The initial image layer state
 * @param getImageDTO A function to retrieve an image DTO from the server, used to update the image source
 */
export const renderIILayer = (
  stage: Konva.Stage,
  layerState: InitialImageLayer,
  getImageDTO: (imageName: string) => Promise<ImageDTO | null>
): void => {
  const konvaLayer = stage.findOne<Konva.Layer>(`#${layerState.id}`) ?? createIILayer(stage, layerState);
  const konvaImage = konvaLayer.findOne<Konva.Image>(`.${INITIAL_IMAGE_LAYER_IMAGE_NAME}`);
  const canvasImageSource = konvaImage?.image();
  let imageSourceNeedsUpdate = false;
  if (canvasImageSource instanceof HTMLImageElement) {
    const image = layerState.image;
    if (image && canvasImageSource.id !== getCALayerImageId(layerState.id, image.name)) {
      imageSourceNeedsUpdate = true;
    } else if (!image) {
      imageSourceNeedsUpdate = true;
    }
  } else if (!canvasImageSource) {
    imageSourceNeedsUpdate = true;
  }

  if (imageSourceNeedsUpdate) {
    updateIILayerImageSource(stage, konvaLayer, layerState, getImageDTO);
  } else if (konvaImage) {
    updateIILayerImageAttrs(stage, konvaImage, layerState);
  }
};
