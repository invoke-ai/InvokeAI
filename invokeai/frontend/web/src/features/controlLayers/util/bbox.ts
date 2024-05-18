import openBase64ImageInTab from 'common/util/openBase64ImageInTab';
import { imageDataToDataURL } from 'features/canvas/util/blobToDataURL';
import { RG_LAYER_OBJECT_GROUP_NAME } from 'features/controlLayers/store/controlLayersSlice';
import Konva from 'konva';
import type { IRect } from 'konva/lib/types';
import { assert } from 'tsafe';

const GET_CLIENT_RECT_CONFIG = { skipTransform: true };

type Extents = {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
};

/**
 * Get the bounding box of an image.
 * @param imageData The ImageData object to get the bounding box of.
 * @returns The minimum and maximum x and y values of the image's bounding box.
 */
const getImageDataBbox = (imageData: ImageData): Extents | null => {
  const { data, width, height } = imageData;
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  let alpha = 0;
  let isEmpty = true;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      alpha = data[(y * width + x) * 4 + 3] ?? 0;
      if (alpha > 0) {
        isEmpty = false;
        if (x < minX) {
          minX = x;
        }
        if (x > maxX) {
          maxX = x;
        }
        if (y < minY) {
          minY = y;
        }
        if (y > maxY) {
          maxY = y;
        }
      }
    }
  }

  return isEmpty ? null : { minX, minY, maxX, maxY };
};

/**
 * Clones a regional guidance konva layer onto an offscreen stage/canvas. This allows the pixel data for a given layer
 * to be captured, manipulated or analyzed without interference from other layers.
 * @param layer The konva layer to clone.
 * @returns The cloned stage and layer.
 */
const getIsolatedRGLayerClone = (layer: Konva.Layer): { stageClone: Konva.Stage; layerClone: Konva.Layer } => {
  const stage = layer.getStage();

  // Construct an offscreen canvas with the same dimensions as the layer's stage.
  const offscreenStageContainer = document.createElement('div');
  const stageClone = new Konva.Stage({
    container: offscreenStageContainer,
    x: stage.x(),
    y: stage.y(),
    width: stage.width(),
    height: stage.height(),
  });

  // Clone the layer and filter out unwanted children.
  const layerClone = layer.clone();
  stageClone.add(layerClone);

  for (const child of layerClone.getChildren()) {
    if (child.name() === RG_LAYER_OBJECT_GROUP_NAME && child.hasChildren()) {
      // We need to cache the group to ensure it composites out eraser strokes correctly
      child.opacity(1);
      child.cache();
    } else {
      // Filter out unwanted children.
      child.destroy();
    }
  }

  return { stageClone, layerClone };
};

/**
 * Get the bounding box of a regional prompt konva layer. This function has special handling for regional prompt layers.
 * @param layer The konva layer to get the bounding box of.
 * @param preview Whether to open a new tab displaying the rendered layer, which is used to calculate the bbox.
 */
export const getLayerBboxPixels = (layer: Konva.Layer, preview: boolean = false): IRect | null => {
  // To calculate the layer's bounding box, we must first export it to a pixel array, then do some math.
  //
  // Though it is relatively fast, we can't use Konva's `getClientRect`. It programmatically determines the rect
  // by calculating the extents of individual shapes from their "vector" shape data.
  //
  // This doesn't work when some shapes are drawn with composite operations that "erase" pixels, like eraser lines.
  // These shapes' extents are still calculated as if they were solid, leading to a bounding box that is too large.
  const { stageClone, layerClone } = getIsolatedRGLayerClone(layer);

  // Get a worst-case rect using the relatively fast `getClientRect`.
  const layerRect = layerClone.getClientRect();
  if (layerRect.width === 0 || layerRect.height === 0) {
    return null;
  }
  // Capture the image data with the above rect.
  const layerImageData = stageClone
    .toCanvas(layerRect)
    .getContext('2d')
    ?.getImageData(0, 0, layerRect.width, layerRect.height);
  assert(layerImageData, "Unable to get layer's image data");

  if (preview) {
    openBase64ImageInTab([{ base64: imageDataToDataURL(layerImageData), caption: layer.id() }]);
  }

  // Calculate the layer's bounding box.
  const layerBbox = getImageDataBbox(layerImageData);

  if (!layerBbox) {
    return null;
  }

  // Correct the bounding box to be relative to the layer's position.
  const correctedLayerBbox = {
    x: layerBbox.minX - Math.floor(stageClone.x()) + layerRect.x - Math.floor(layer.x()),
    y: layerBbox.minY - Math.floor(stageClone.y()) + layerRect.y - Math.floor(layer.y()),
    width: layerBbox.maxX - layerBbox.minX,
    height: layerBbox.maxY - layerBbox.minY,
  };

  return correctedLayerBbox;
};

/**
 * Get the bounding box of a konva layer. This function is faster than `getLayerBboxPixels` but less accurate. It
 * should only be used when there are no eraser strokes or shapes in the layer.
 * @param layer The konva layer to get the bounding box of.
 * @returns The bounding box of the layer.
 */
export const getLayerBboxFast = (layer: Konva.Layer): IRect => {
  const bbox = layer.getClientRect(GET_CLIENT_RECT_CONFIG);
  return {
    x: Math.floor(bbox.x),
    y: Math.floor(bbox.y),
    width: Math.floor(bbox.width),
    height: Math.floor(bbox.height),
  };
};
