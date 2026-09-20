import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
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
  const canvasManager = useCanvasManager();
  const pathTool = canvasManager.tool.tools.path;
  const editSession = useStore(pathTool.$editSession);

  const mayUndoCanvas = useAppSelector(selectCanvasMayUndo);
  const mayUndo = editSession ? pathTool.canUndoEditSession() : mayUndoCanvas;
  const handleUndo = useCallback(() => {
    if (pathTool.hasActiveEditSession()) {
      pathTool.undoEditSession();
      return;
    }
    dispatch(canvasUndo());
  }, [dispatch, pathTool]);
  useRegisteredHotkeys({
    id: 'undo',
    category: 'canvas',
    callback: handleUndo,
    options: { enabled: mayUndo && !isBusy, preventDefault: true },
    dependencies: [mayUndo, isBusy, handleUndo],
  });

  const mayRedoCanvas = useAppSelector(selectCanvasMayRedo);
  const mayRedo = editSession ? pathTool.canRedoEditSession() : mayRedoCanvas;
  const handleRedo = useCallback(() => {
    if (pathTool.hasActiveEditSession()) {
      pathTool.redoEditSession();
      return;
    }
    dispatch(canvasRedo());
  }, [dispatch, pathTool]);
  useRegisteredHotkeys({
    id: 'redo',
    category: 'canvas',
    callback: handleRedo,
    options: { enabled: mayRedo && !isBusy, preventDefault: true },
    dependencies: [mayRedo, handleRedo, isBusy],
  });
};
