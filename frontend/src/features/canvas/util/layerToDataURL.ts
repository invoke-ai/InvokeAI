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
        x: boundingBox.x + stageCoordinates.x,
        y: boundingBox.y + stageCoordinates.y,
        width: boundingBox.width,
        height: boundingBox.height,
      }
    : {
        x: x,
        y: y,
        width: width,
        height: height,
      };

  const dataURL = layer.toDataURL(dataURLBoundingBox);

  // Unscale the canvas
  layer.scale(tempScale);

  return {
    dataURL,
    boundingBox: {
      x: relativeClientRect.x,
      y: relativeClientRect.y,
      width: width,
      height: height,
    },
  };
};

export default layerToDataURL;
