import openBase64ImageInTab from 'common/util/openBase64ImageInTab';
import { imageDataToDataURL } from 'features/canvas/util/blobToDataURL';
import Konva from 'konva';
import type { Layer as KonvaLayerType } from 'konva/lib/Layer';
import type { Node as KonvaNodeType, NodeConfig as KonvaNodeConfigType } from 'konva/lib/Node';
import type { IRect } from 'konva/lib/types';
import { assert } from 'tsafe';

/**
 * Get the bounding box of an image.
 * @param imageData The ImageData object to get the bounding box of.
 * @returns The minimum and maximum x and y values of the image's bounding box.
 */
const getImageDataBbox = (imageData: ImageData) => {
  const { data, width, height } = imageData;
  let minX = width;
  let minY = height;
  let maxX = 0;
  let maxY = 0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const alpha = data[(y * width + x) * 4 + 3] ?? 0;
      if (alpha > 0) {
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
      }
    }
  }

  return { minX, minY, maxX, maxY };
};

/**
 * Get the bounding box of a regional prompt konva layer. This function has special handling for regional prompt layers.
 * @param layer The konva layer to get the bounding box of.
 * @param filterChildren Optional filter function to exclude certain children from the bounding box calculation. Defaults to including all children.
 * @param preview Whether to open a new tab displaying the rendered layer, which is used to calculate the bbox.
 */
export const getKonvaLayerBbox = (
  layer: KonvaLayerType,
  filterChildren?: (item: KonvaNodeType<KonvaNodeConfigType>) => boolean,
  preview: boolean = false
): IRect => {
  // To calculate the layer's bounding box, we must first export it to a pixel array, then do some math.
  //
  // Though it is relatively fast, we can't use Konva's `getClientRect`. It programmatically determines the rect
  // by calculating the extents of individual shapes from their "vector" shape data.
  //
  // This doesn't work when some shapes are drawn with composite operations that "erase" pixels, like eraser lines.
  // These shapes' extents are still calculated as if they were solid, leading to a bounding box that is too large.

  const stage = layer.getStage();

  // Construct and offscreen canvas on which we will do the bbox calculations.
  const offscreenStageContainer = document.createElement('div');
  const offscreenStage = new Konva.Stage({
    container: offscreenStageContainer,
    width: stage.width(),
    height: stage.height(),
  });

  // Clone the layer and filter out unwanted children.
  const layerClone = layer.clone();
  offscreenStage.add(layerClone);

  if (filterChildren) {
    for (const child of layerClone.getChildren(filterChildren)) {
      child.destroy();
    }
  }

  // Get a worst-case rect using the relatively fast `getClientRect`.
  const layerRect = layerClone.getClientRect();

  // Capture the image data with the above rect.
  const layerImageData = offscreenStage
    .toCanvas(layerRect)
    .getContext('2d')
    ?.getImageData(0, 0, layerRect.width, layerRect.height);
  assert(layerImageData, "Unable to get layer's image data");

  if (preview) {
    openBase64ImageInTab([{ base64: imageDataToDataURL(layerImageData), caption: layer.id() }]);
  }

  // Calculate the layer's bounding box.
  const layerBbox = getImageDataBbox(layerImageData);

  // Correct the bounding box to be relative to the layer's position.
  const correctedLayerBbox = {
    x: layerBbox.minX - stage.x() + layerRect.x - layer.x(),
    y: layerBbox.minY - stage.y() + layerRect.y - layer.y(),
    width: layerBbox.maxX - layerBbox.minX,
    height: layerBbox.maxY - layerBbox.minY,
  };

  return correctedLayerBbox;
};
