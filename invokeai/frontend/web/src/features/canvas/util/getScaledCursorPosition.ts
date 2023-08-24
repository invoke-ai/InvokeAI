import { Stage } from 'konva/lib/Stage';

const getScaledCursorPosition = (stage: Stage) => {
  const pointerPosition = stage.getPointerPosition();

  const stageTransform = stage.getAbsoluteTransform().copy();

  if (!pointerPosition || !stageTransform) {
    return;
  }

  const scaledCursorPosition = stageTransform.invert().point(pointerPosition);

  return {
    x: scaledCursorPosition.x,
    y: scaledCursorPosition.y,
  };
};

export default getScaledCursorPosition;
