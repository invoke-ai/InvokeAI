import { useAppDispatch } from 'app/storeHooks';
import { useCallback } from 'react';
import { mouseLeftCanvas } from 'features/canvas/store/canvasSlice';

const useCanvasMouseOut = () => {
  const dispatch = useAppDispatch();

  return useCallback(() => {
    dispatch(mouseLeftCanvas());
  }, [dispatch]);
};

export default useCanvasMouseOut;
