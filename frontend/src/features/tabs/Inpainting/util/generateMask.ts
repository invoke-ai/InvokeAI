import Konva from 'konva';
import { IRect } from 'konva/lib/types';
import { MaskLine } from '../inpaintingSlice';

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
) => {
  const { x, y, width, height } = boundingBox;

  const offscreenContainer = document.createElement('div');

  const stage = new Konva.Stage({
    container: offscreenContainer,
    width: image.width,
    height: image.height,
  });

  const layer = new Konva.Layer();

  stage.add(layer);

  lines.forEach((line) =>
    layer.add(
      new Konva.Line({
        points: line.points,
        stroke: 'rgb(255,255,255)',
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

  // check if mask is empty
  const pixelBuffer = new Uint32Array(
    layer.getContext().getImageData(x, y, width, height).data.buffer
  );

  const isMaskEmpty = !pixelBuffer.some((color) => color !== 0);

  if (isMaskEmpty) {
    layer.add(
      new Konva.Rect({
        ...boundingBox,
        fill: 'rgb(0,0,0)',
      })
    );
  }

  layer.add(
    new Konva.Image({ image: image, globalCompositeOperation: 'source-out' })
  );

  const maskDataURL = stage.toDataURL();

  offscreenContainer.remove();

  return { maskDataURL, isMaskEmpty };
};

export default generateMask;
