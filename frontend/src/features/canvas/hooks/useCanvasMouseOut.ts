import { useAppDispatch } from 'app/store';
import { useCallback } from 'react';
import { setCursorPosition, setIsDrawing } from 'features/canvas/store/canvasSlice';

const useCanvasMouseOut = () => {
  const dispatch = useAppDispatch();

  return useCallback(() => {
    dispatch(setCursorPosition(null));
    dispatch(setIsDrawing(false));
  }, [dispatch]);
};

export default useCanvasMouseOut;
