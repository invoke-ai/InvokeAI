import { Vector2d } from 'konva/lib/types';

const calculateCoordinates = (
  containerWidth: number,
  containerHeight: number,
  containerX: number,
  containerY: number,
  contentWidth: number,
  contentHeight: number,
  scale: number
): Vector2d => {
  const x = containerWidth / 2 - (containerX + contentWidth / 2) * scale;
  const y = containerHeight / 2 - (containerY + contentHeight / 2) * scale;
  return { x, y };
};

export default calculateCoordinates;
