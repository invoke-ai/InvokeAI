import Konva from 'konva';
import { IRect } from 'konva/lib/types';

const layerToDataURL = (
  layer: Konva.Layer,
  stageScale: number,
  boundingBox?: IRect
) => {
  const tempScale = layer.scale();

  const relativeClientRect = layer.getClientRect({
    relativeTo: layer.getParent(),
  });

  // Scale the canvas before getting it as a Blob
  layer.scale({
    x: 1 / stageScale,
    y: 1 / stageScale,
  });

  const { x, y, width, height } = layer.getClientRect();

  const scaledBoundingBox = boundingBox
    ? {
        x: Math.round(boundingBox.x / stageScale),
        y: Math.round(boundingBox.y / stageScale),
        width: Math.round(boundingBox.width / stageScale),
        height: Math.round(boundingBox.height / stageScale),
      }
    : {
        x: Math.round(x),
        y: Math.round(y),
        width: Math.round(width),
        height: Math.round(height),
      };

  const dataURL = layer.toDataURL(scaledBoundingBox);

  // Unscale the canvas
  layer.scale(tempScale);

  return {
    dataURL,
    boundingBox: {
      x: Math.round(relativeClientRect.x),
      y: Math.round(relativeClientRect.y),
      width: Math.round(width),
      height: Math.round(height),
    },
  };
};

export default layerToDataURL;
