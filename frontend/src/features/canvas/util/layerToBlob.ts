import Konva from 'konva';

const layerToBlob = async (layer: Konva.Layer, stageScale: number) => {
  const tempScale = layer.scale();

  const { x: relativeX, y: relativeY } = layer.getClientRect({
    relativeTo: layer.getParent(),
  });

  // Scale the canvas before getting it as a Blob
  layer.scale({
    x: 1 / stageScale,
    y: 1 / stageScale,
  });

  const clientRect = layer.getClientRect();

  const blob = await layer.toBlob(clientRect);

  // Unscale the canvas
  layer.scale(tempScale);

  return { blob, relativeX, relativeY };
};

export default layerToBlob;
