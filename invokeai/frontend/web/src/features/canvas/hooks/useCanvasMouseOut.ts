import { setCanvasInteractionStateMouseOut } from 'features/canvas/store/canvasNanostore';
import { useCallback } from 'react';

const useCanvasMouseOut = () => {
  const onMouseOut = useCallback(() => {
    setCanvasInteractionStateMouseOut();
  }, []);

  return onMouseOut;
};

export default useCanvasMouseOut;
