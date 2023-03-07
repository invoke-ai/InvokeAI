import { useState } from 'react';

const useImageTransform = () => {
  const [rotation, setRotation] = useState(0);
  const [scaleX, setScaleX] = useState(1);
  const [scaleY, setScaleY] = useState(1);

  const rotateCounterClockwise = () => {
    if (rotation === -270) {
      setRotation(0);
    } else {
      setRotation(rotation - 90);
    }
  };

  const rotateClockwise = () => {
    if (rotation === 270) {
      setRotation(0);
    } else {
      setRotation(rotation + 90);
    }
  };

  const flipHorizontally = () => {
    setScaleX(scaleX * -1);
  };

  const flipVertically = () => {
    setScaleY(scaleY * -1);
  };

  const reset = () => {
    setRotation(0);
    setScaleX(1);
    setScaleY(1);
  };

  return {
    rotation,
    scaleX,
    scaleY,
    flipHorizontally,
    flipVertically,
    rotateCounterClockwise,
    rotateClockwise,
    reset,
  };
};

export default useImageTransform;
