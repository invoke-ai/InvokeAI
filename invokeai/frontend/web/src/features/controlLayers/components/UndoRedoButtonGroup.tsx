/* eslint-disable i18next/no-literal-string */
import { ButtonGroup, IconButton } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiArrowClockwiseBold, PiArrowCounterClockwiseBold } from 'react-icons/pi';
import { useDispatch } from 'react-redux';
import { ActionCreators } from 'redux-undo';

export const UndoRedoButtonGroup = memo(() => {
  const { t } = useTranslation();
  const dispatch = useDispatch();

  const mayUndo = useAppSelector(() => true);
  const handleUndo = useCallback(() => {
    // TODO(psyche): Implement undo
    dispatch(ActionCreators.undo());
  }, [dispatch]);
  useHotkeys(['meta+z', 'ctrl+z'], handleUndo, { enabled: mayUndo, preventDefault: true }, [mayUndo, handleUndo]);

  const mayRedo = useAppSelector(() => true);
  const handleRedo = useCallback(() => {
    // TODO(psyche): Implement redo
    dispatch(ActionCreators.redo());
  }, [dispatch]);
  useHotkeys(['meta+shift+z', 'ctrl+shift+z'], handleRedo, { enabled: mayRedo, preventDefault: true }, [
    mayRedo,
    handleRedo,
  ]);

  return (
    <ButtonGroup>
      <IconButton
        aria-label={t('unifiedCanvas.undo')}
        tooltip={t('unifiedCanvas.undo')}
        onClick={handleUndo}
        icon={<PiArrowCounterClockwiseBold />}
        isDisabled={!mayUndo}
      />
      <IconButton
        aria-label={t('unifiedCanvas.redo')}
        tooltip={t('unifiedCanvas.redo')}
        onClick={handleRedo}
        icon={<PiArrowClockwiseBold />}
        isDisabled={!mayRedo}
      />
    </ButtonGroup>
  );
});

UndoRedoButtonGroup.displayName = 'UndoRedoButtonGroup';
