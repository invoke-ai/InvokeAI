import Konva from 'konva';
import { IRect, Vector2d } from 'konva/lib/types';

const layerToDataURL = (
  layer: Konva.Layer,
  stageScale: number,
  stageCoordinates: Vector2d,
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
  const dataURLBoundingBox = boundingBox
    ? {
        x: Math.round(boundingBox.x + stageCoordinates.x),
        y: Math.round(boundingBox.y + stageCoordinates.y),
        width: Math.round(boundingBox.width),
        height: Math.round(boundingBox.height),
      }
    : {
        x: Math.round(x),
        y: Math.round(y),
        width: Math.round(width),
        height: Math.round(height),
      };

  const dataURL = layer.toDataURL(dataURLBoundingBox);

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
