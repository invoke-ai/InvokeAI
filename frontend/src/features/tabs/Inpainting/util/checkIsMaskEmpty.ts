import Konva from 'konva';
import { MaskLine } from '../inpaintingSlice';

/**
 * Converts canvas into pixel buffer and checks if it is empty (all pixels full alpha).
 */
const checkIsMaskEmpty = (image: HTMLImageElement, lines: MaskLine[]) => {
  const offscreenContainer = document.createElement('div');

  const { width, height } = image;

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

  offscreenContainer.remove();

  const pixelBuffer = new Uint32Array(
    layer.getContext().getImageData(0, 0, width, height).data.buffer
  );

  return !pixelBuffer.some((color) => color !== 0);
};

export default checkIsMaskEmpty;
