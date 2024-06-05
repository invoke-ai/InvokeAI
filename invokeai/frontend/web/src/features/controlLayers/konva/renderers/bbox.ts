import openBase64ImageInTab from 'common/util/openBase64ImageInTab';
import { imageDataToDataURL } from 'features/canvas/util/blobToDataURL';
import { BBOX_SELECTED_STROKE } from 'features/controlLayers/konva/constants';
import { getLayerBboxId, LAYER_BBOX_NAME, RG_LAYER_OBJECT_GROUP_NAME } from 'features/controlLayers/konva/naming';
import type { Layer, Tool } from 'features/controlLayers/store/types';
import { isRegionalGuidanceLayer } from 'features/controlLayers/store/types';
import Konva from 'konva';
import type { IRect } from 'konva/lib/types';
import { assert } from 'tsafe';

/**
 * Logic to create and render bounding boxes for layers.
 * Some utils are included for calculating bounding boxes.
 */

type Extents = {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
};

const GET_CLIENT_RECT_CONFIG = { skipTransform: true };

/**
 * Get the bounding box of an image.
 * @param imageData The ImageData object to get the bounding box of.
 * @returns The minimum and maximum x and y values of the image's bounding box, or null if the image has no pixels.
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
const getLayerBboxPixels = (layer: Konva.Layer, preview: boolean = false): IRect | null => {
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

/**
 * Creates a bounding box rect for a layer.
 * @param layerState The layer state for the layer to create the bounding box for
 * @param konvaLayer The konva layer to attach the bounding box to
 */
const createBboxRect = (layerState: Layer, konvaLayer: Konva.Layer): Konva.Rect => {
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
 * Calculates the bbox of each regional guidance layer. Only calculates if the mask has changed.
 * @param stage The konva stage
 * @param layerStates An array of layers to calculate bboxes for
 * @param onBboxChanged Callback for when the bounding box changes
 */
export const updateBboxes = (
  stage: Konva.Stage,
  layerStates: Layer[],
  onBboxChanged: (layerId: string, bbox: IRect | null) => void
): void => {
  for (const rgLayer of layerStates.filter(isRegionalGuidanceLayer)) {
    const konvaLayer = stage.findOne<Konva.Layer>(`#${rgLayer.id}`);
    assert(konvaLayer, `Layer ${rgLayer.id} not found in stage`);
    // We only need to recalculate the bbox if the layer has changed
    if (rgLayer.bboxNeedsUpdate) {
      const bboxRect = konvaLayer.findOne<Konva.Rect>(`.${LAYER_BBOX_NAME}`) ?? createBboxRect(rgLayer, konvaLayer);

      // Hide the bbox while we calculate the new bbox, else the bbox will be included in the calculation
      const visible = bboxRect.visible();
      bboxRect.visible(false);

      if (rgLayer.objects.length === 0) {
        // No objects - no bbox to calculate
        onBboxChanged(rgLayer.id, null);
      } else {
        // Calculate the bbox by rendering the layer and checking its pixels
        onBboxChanged(rgLayer.id, getLayerBboxPixels(konvaLayer));
      }

      // Restore the visibility of the bbox
      bboxRect.visible(visible);
    }
  }
};

/**
 * Renders the bounding boxes for the layers.
 * @param stage The konva stage
 * @param layerStates An array of layers to draw bboxes for
 * @param tool The current tool
 * @returns
 */
export const renderBboxes = (stage: Konva.Stage, layerStates: Layer[], tool: Tool): void => {
  // Hide all bboxes so they don't interfere with getClientRect
  for (const bboxRect of stage.find<Konva.Rect>(`.${LAYER_BBOX_NAME}`)) {
    bboxRect.visible(false);
    bboxRect.listening(false);
  }
  // No selected layer or not using the move tool - nothing more to do here
  if (tool !== 'move') {
    return;
  }

  for (const layer of layerStates.filter(isRegionalGuidanceLayer)) {
    if (!layer.bbox) {
      continue;
    }
    const konvaLayer = stage.findOne<Konva.Layer>(`#${layer.id}`);
    assert(konvaLayer, `Layer ${layer.id} not found in stage`);

    const bboxRect = konvaLayer.findOne<Konva.Rect>(`.${LAYER_BBOX_NAME}`) ?? createBboxRect(layer, konvaLayer);

    bboxRect.setAttrs({
      visible: !layer.bboxNeedsUpdate,
      listening: layer.isSelected,
      x: layer.bbox.x,
      y: layer.bbox.y,
      width: layer.bbox.width,
      height: layer.bbox.height,
      stroke: layer.isSelected ? BBOX_SELECTED_STROKE : '',
    });
  }
};
