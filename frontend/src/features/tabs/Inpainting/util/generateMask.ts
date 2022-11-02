import Konva from 'konva';
import { IRect } from 'konva/lib/types';
import { MaskLine } from '../inpaintingSlice';

/**
 * Re-draws the mask canvas onto a new Konva stage.
 */
export const generateMaskCanvas = (
  image: HTMLImageElement,
  lines: MaskLine[]
): {
  stage: Konva.Stage;
  layer: Konva.Layer;
} => {
  const { width, height } = image;

  const offscreenContainer = document.createElement('div');

  const stage = new Konva.Stage({
    container: offscreenContainer,
    width: width,
    height: height,
  });

  const layer = new Konva.Layer();

  stage.add(layer);

  lines.forEach((line) =>
    layer.add(
      new Konva.Line({
        points: line.points,
        stroke: 'rgb(0,0,0)',
        strokeWidth: line.strokeWidth * 2,
        tension: 0,
        lineCap: 'round',
        lineJoin: 'round',
        shadowForStrokeEnabled: false,
        globalCompositeOperation:
          line.tool === 'brush' ? 'source-over' : 'destination-out',
      })
    )
  );

  layer.draw();

  offscreenContainer.remove();

  return { stage, layer };
};

/**
 * Check if the bounding box region has only fully transparent pixels.
 */
export const checkIsRegionEmpty = (
  stage: Konva.Stage,
  boundingBox: IRect
): boolean => {
  const imageData = stage
    .toCanvas()
    .getContext('2d')
    ?.getImageData(
      boundingBox.x,
      boundingBox.y,
      boundingBox.width,
      boundingBox.height
    );

  if (!imageData) {
    throw new Error('Unable to get image data from generated canvas');
  }

  const pixelBuffer = new Uint32Array(imageData.data.buffer);

  return !pixelBuffer.some((color) => color !== 0);
};

/**
 * Generating a mask image from InpaintingCanvas.tsx is not as simple
 * as calling toDataURL() on the canvas, because the mask may be represented
 * by colored lines or transparency, or the user may have inverted the mask
 * display.
 *
 * So we need to regenerate the mask image by creating an offscreen canvas,
 * drawing the mask and compositing everything correctly to output a valid
 * mask image.
 */
const generateMask = (
  image: HTMLImageElement,
  lines: MaskLine[],
  boundingBox: IRect
): { maskDataURL: string; isMaskEmpty: boolean } => {
  // create an offscreen canvas and add the mask to it
  const { stage, layer } = generateMaskCanvas(image, lines);

  // check if the mask layer is empty
  const isMaskEmpty = checkIsRegionEmpty(stage, boundingBox);

  // composite the image onto the mask layer
  layer.add(
    new Konva.Image({ image: image, globalCompositeOperation: 'source-out' })
  );

  const maskDataURL = stage.toDataURL({ ...boundingBox });

  return { maskDataURL, isMaskEmpty };
};

export default generateMask;
