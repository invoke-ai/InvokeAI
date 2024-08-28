/* eslint-disable i18next/no-literal-string */
import { ButtonGroup, IconButton } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { canvasRedo, canvasUndo } from 'features/controlLayers/store/canvasSlice';
import { selectCanvasMayRedo, selectCanvasMayUndo } from 'features/controlLayers/store/selectors';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiArrowClockwiseBold, PiArrowCounterClockwiseBold } from 'react-icons/pi';
import { useDispatch } from 'react-redux';

export const UndoRedoButtonGroup = memo(() => {
  const { t } = useTranslation();
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

  return (
    <ButtonGroup isAttached={false}>
      <IconButton
        aria-label={t('unifiedCanvas.undo')}
        tooltip={t('unifiedCanvas.undo')}
        onClick={handleUndo}
        icon={<PiArrowCounterClockwiseBold />}
        isDisabled={!mayUndo}
        variant="ghost"
      />
      <IconButton
        aria-label={t('unifiedCanvas.redo')}
        tooltip={t('unifiedCanvas.redo')}
        onClick={handleRedo}
        icon={<PiArrowClockwiseBold />}
        isDisabled={!mayRedo}
        variant="ghost"
      />
    </ButtonGroup>
  );
});

UndoRedoButtonGroup.displayName = 'UndoRedoButtonGroup';
