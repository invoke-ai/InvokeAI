import { useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { canvasRedo, canvasUndo } from 'features/controlLayers/store/canvasSlice';
import { selectCanvasMayRedo, selectCanvasMayUndo } from 'features/controlLayers/store/selectors';
import { useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useDispatch } from 'react-redux';

export const useCanvasUndoRedoHotkeys = () => {
  useAssertSingleton('useCanvasUndoRedo');
  const dispatch = useDispatch();

  const mayUndo = useAppSelector(selectCanvasMayUndo);
  const handleUndo = useCallback(() => {
    dispatch(canvasUndo());
  }, [dispatch]);
  useHotkeys(['meta+z', 'ctrl+z'], handleUndo, { enabled: mayUndo, preventDefault: true }, [mayUndo, handleUndo]);

  const mayRedo = useAppSelector(selectCanvasMayRedo);
  const handleRedo = useCallback(() => {
    dispatch(canvasRedo());
  }, [dispatch]);
  useHotkeys(['meta+shift+z', 'ctrl+shift+z'], handleRedo, { enabled: mayRedo, preventDefault: true }, [
    mayRedo,
    handleRedo,
  ]);
};
