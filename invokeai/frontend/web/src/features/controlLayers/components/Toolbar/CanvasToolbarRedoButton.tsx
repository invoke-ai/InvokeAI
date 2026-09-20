import { IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { canvasRedo } from 'features/controlLayers/store/canvasSlice';
import { selectCanvasMayRedo } from 'features/controlLayers/store/selectors';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowClockwiseBold } from 'react-icons/pi';

export const CanvasToolbarRedoButton = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isBusy = useCanvasIsBusy();
  const canvasManager = useCanvasManager();
  const pathTool = canvasManager.tool.tools.path;
  const editSession = useStore(pathTool.$editSession);
  const mayRedoCanvas = useAppSelector(selectCanvasMayRedo);
  const mayRedo = editSession ? pathTool.canRedoEditSession() : mayRedoCanvas;
  const onClick = useCallback(() => {
    if (pathTool.hasActiveEditSession()) {
      pathTool.redoEditSession();
      return;
    }
    dispatch(canvasRedo());
  }, [dispatch, pathTool]);

  return (
    <IconButton
      aria-label={t('hotkeys.canvas.redo.title')}
      tooltip={t('hotkeys.canvas.redo.title')}
      onClick={onClick}
      icon={<PiArrowClockwiseBold />}
      variant="link"
      alignSelf="stretch"
      isDisabled={isBusy || !mayRedo}
    />
  );
});

CanvasToolbarRedoButton.displayName = 'CanvasToolbarRedoButton';
