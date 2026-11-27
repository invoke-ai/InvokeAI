import { useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { canvasRedo, canvasUndo } from 'features/controlLayers/store/canvasSlice';
import { selectCanvasMayRedo, selectCanvasMayUndo } from 'features/controlLayers/store/selectors';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { useCallback } from 'react';
import { useDispatch } from 'react-redux';

export const useCanvasUndoRedoHotkeys = () => {
  useAssertSingleton('useCanvasUndoRedo');
  const dispatch = useDispatch();
  const isBusy = useCanvasIsBusy();

  const mayUndo = useAppSelector(selectCanvasMayUndo);
  const handleUndo = useCallback(() => {
    dispatch(canvasUndo());
  }, [dispatch]);
  useRegisteredHotkeys({
    id: 'undo',
    category: 'canvas',
    callback: handleUndo,
    options: { enabled: mayUndo && !isBusy, preventDefault: true },
    dependencies: [mayUndo, isBusy, handleUndo],
  });

  const mayRedo = useAppSelector(selectCanvasMayRedo);
  const handleRedo = useCallback(() => {
    dispatch(canvasRedo());
  }, [dispatch]);
  useRegisteredHotkeys({
    id: 'redo',
    category: 'canvas',
    callback: handleRedo,
    options: { enabled: mayRedo && !isBusy, preventDefault: true },
    dependencies: [mayRedo, handleRedo, isBusy],
  });
};
