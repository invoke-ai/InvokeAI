import { IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { canvasUndo } from 'features/controlLayers/store/canvasSlice';
import { selectCanvasMayUndo } from 'features/controlLayers/store/selectors';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';

export const CanvasToolbarUndoButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isBusy = useCanvasIsBusy();
  const canvasManager = useCanvasManager();
  const pathTool = canvasManager.tool.tools.path;
  const editSession = useStore(pathTool.$editSession);
  const mayUndoCanvas = useAppSelector(selectCanvasMayUndo);
  const mayUndo = editSession ? pathTool.canUndoEditSession() : mayUndoCanvas;
  const onClick = useCallback(() => {
    if (pathTool.hasActiveEditSession()) {
      pathTool.undoEditSession();
      return;
    }
    dispatch(canvasUndo());
  }, [dispatch, pathTool]);

  return (
    <IconButton
      aria-label={t('hotkeys.canvas.undo.title')}
      tooltip={t('hotkeys.canvas.undo.title')}
      onClick={onClick}
      icon={<PiArrowCounterClockwiseBold />}
      variant="link"
      alignSelf="stretch"
      isDisabled={isBusy || !mayUndo}
    />
  );
});

CanvasToolbarUndoButton.displayName = 'CanvasToolbarUndoButton';
