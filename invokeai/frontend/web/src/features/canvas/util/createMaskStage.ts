import type { CanvasMaskLine } from 'features/canvas/store/canvasTypes';
import Konva from 'konva';
import type { IRect } from 'konva/lib/types';

/**
 * Creates a stage from array of mask objects.
 * We cannot just convert the mask layer to a blob because it uses a texture with transparent areas.
 * So instead we create a new stage with the mask layer and composite it onto a white background.
 */
const createMaskStage = async (
  lines: CanvasMaskLine[],
  boundingBox: IRect,
  shouldInvertMask: boolean
): Promise<Konva.Stage> => {
  // create an offscreen canvas and add the mask to it
  const { width, height } = boundingBox;

  const offscreenContainer = document.createElement('div');

  const maskStage = new Konva.Stage({
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
      fill: shouldInvertMask ? 'black' : 'white',
    })
  );

  lines.forEach((line) =>
    maskLayer.add(
      new Konva.Line({
        points: line.points,
        stroke: shouldInvertMask ? 'white' : 'black',
        strokeWidth: line.strokeWidth * 2,
        tension: 0,
        lineCap: 'round',
        lineJoin: 'round',
        shadowForStrokeEnabled: false,
        globalCompositeOperation: line.tool === 'brush' ? 'source-over' : 'destination-out',
      })
    )
  );

  maskStage.add(baseLayer);
  maskStage.add(maskLayer);

  // you'd think we can't do this until we finish with the maskStage, but we can
  offscreenContainer.remove();

  return maskStage;
};

export default createMaskStage;
