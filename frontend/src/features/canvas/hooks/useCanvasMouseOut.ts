import { useAppDispatch } from 'app/store';
import _ from 'lodash';
import { useCallback } from 'react';
import { setCursorPosition, setIsDrawing } from '../canvasSlice';

const useCanvasMouseOut = () => {
  const dispatch = useAppDispatch();

  return useCallback(() => {
    dispatch(setCursorPosition(null));
    dispatch(setIsDrawing(false));
  }, [dispatch]);
};

export default useCanvasMouseOut;
