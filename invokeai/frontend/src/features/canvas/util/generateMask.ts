import { CanvasMaskLine } from 'features/canvas/store/canvasTypes';
import Konva from 'konva';
import { IRect } from 'konva/lib/types';

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
const generateMask = (lines: CanvasMaskLine[], boundingBox: IRect): string => {
  // create an offscreen canvas and add the mask to it
  const { width, height } = boundingBox;

  const offscreenContainer = document.createElement('div');

  const stage = new Konva.Stage({
    container: offscreenContainer,
    width: width,
    height: height,
  });

  const baseLayer = new Konva.Layer();
  const maskLayer = new Konva.Layer();

  // composite the image onto the mask layer
  baseLayer.add(
    new Konva.Rect({
      ...boundingBox,
      fill: 'white',
    })
  );

  lines.forEach((line) =>
    maskLayer.add(
      new Konva.Line({
        points: line.points,
        stroke: 'black',
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

  stage.add(baseLayer);
  stage.add(maskLayer);

  const dataURL = stage.toDataURL({ ...boundingBox });

  offscreenContainer.remove();

  return dataURL;
};

export default generateMask;
