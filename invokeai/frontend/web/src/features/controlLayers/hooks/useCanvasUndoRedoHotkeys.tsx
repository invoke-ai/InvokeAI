import { useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { canvasRedo, canvasUndo } from 'features/controlLayers/store/canvasSlice';
import { selectCanvasMayRedo, selectCanvasMayUndo } from 'features/controlLayers/store/selectors';
import { useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useDispatch } from 'react-redux';

export const useCanvasUndoRedoHotkeys = () => {
  useAssertSingleton('useCanvasUndoRedo');
  const dispatch = useDispatch();
  const isBusy = useCanvasIsBusy();

  const mayUndo = useAppSelector(selectCanvasMayUndo);
  const handleUndo = useCallback(() => {
    dispatch(canvasUndo());
  }, [dispatch]);
  useHotkeys(['meta+z', 'ctrl+z'], handleUndo, { enabled: mayUndo && !isBusy, preventDefault: true }, [
    mayUndo,
    isBusy,
    handleUndo,
  ]);

  const mayRedo = useAppSelector(selectCanvasMayRedo);
  const handleRedo = useCallback(() => {
    dispatch(canvasRedo());
  }, [dispatch]);
  useHotkeys(
    ['meta+shift+z', 'ctrl+shift+z', 'meta+y', 'ctrl+y'],
    handleRedo,
    { enabled: mayRedo && !isBusy, preventDefault: true },
    [mayRedo, handleRedo, isBusy]
  );
};
