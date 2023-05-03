import { useAppDispatch } from 'app/store/storeHooks';
import { mouseLeftCanvas } from 'features/canvas/store/canvasSlice';
import { useCallback } from 'react';

const useCanvasMouseOut = () => {
  const dispatch = useAppDispatch();

  return useCallback(() => {
    dispatch(mouseLeftCanvas());
  }, [dispatch]);
};

export default useCanvasMouseOut;
