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
  // To calculate the layer's bounding box, we must first render it to a pixel array, then do some math.
  // We can't use konva's `layer.getClientRect()`, because this includes all shapes, not just visible ones.
  // That would include eraser strokes, and the resultant bbox would be too large.
  const stage = layer.getStage();

  // Construct and offscreen canvas and add just the layer to it.
  const offscreenStageContainer = document.createElement('div');
  const offscreenStage = new Konva.Stage({
    container: offscreenStageContainer,
    width: stage.width(),
    height: stage.height(),
  });

  // Clone the layer and filter out unwanted children.
  // TODO: Would be more efficient to create a totally new layer and add only the children we want, but possibly less
  // accurate, as we wouldn't get the original layer's config and such.
  const layerClone = layer.clone();
  offscreenStage.add(layerClone);

  for (const child of layerClone.getChildren()) {
    if (filterChildren && filterChildren(child)) {
      child.destroy();
    } else {
      // We need to re-cache to handle children with transparency and multiple objects - like prompt region layers.
      // child.cache();
    }
  }

  // Get the layer's image data, ensuring we capture an area large enough to include the full layer, including any
  // portions that are outside the current stage bounds.
  const layerRect = layerClone.getClientRect();

  // Render the canvas, large enough to capture the full layer.
  const x = -layerRect.width; // start from left of layer, as far left as the layer might be
  const y = -layerRect.height; // start from top of layer, as far up as the layer might be
  const width = stage.width() + layerRect.width * 2; // stage width + layer width on left/right
  const height = stage.height() + layerRect.height * 2; // stage height + layer height on top/bottom

  // Capture the image data with the above rect.
  const layerImageData = offscreenStage
    .toCanvas({ x, y, width, height })
    .getContext('2d')
    ?.getImageData(0, 0, width, height);
  assert(layerImageData, "Unable to get layer's image data");

  if (preview) {
    openBase64ImageInTab([{ base64: imageDataToDataURL(layerImageData), caption: layer.id() }]);
  }

  // Calculate the layer's bounding box.
  const layerBbox = getImageDataBbox(layerImageData);

  // Correct the bounding box to be relative to the layer's position.
  const correctedLayerBbox = {
    x: layerBbox.minX - layerRect.width - layer.x(),
    y: layerBbox.minY - layerRect.height - layer.y(),
    width: layerBbox.maxX - layerBbox.minX,
    height: layerBbox.maxY - layerBbox.minY,
  };

  return correctedLayerBbox;
};
