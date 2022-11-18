import Konva from 'konva';

const layerToDataURL = (layer: Konva.Layer, stageScale: number) => {
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

  const dataURL = layer.toDataURL({
    x: Math.round(x),
    y: Math.round(y),
    width: Math.round(width),
    height: Math.round(height),
  });

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
