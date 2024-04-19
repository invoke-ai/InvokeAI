/* eslint-disable i18next/no-literal-string */
import { ButtonGroup, IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { redoRegionalPrompts, undoRegionalPrompts } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiArrowClockwiseBold, PiArrowCounterClockwiseBold } from 'react-icons/pi';

export const UndoRedoButtonGroup = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const mayUndo = useAppSelector((s) => s.regionalPrompts.past.length > 0);
  const undo = useCallback(() => {
    dispatch(undoRegionalPrompts());
  }, [dispatch]);
  useHotkeys(['meta+z', 'ctrl+z'], undo, { enabled: mayUndo, preventDefault: true }, [mayUndo, undo]);

  const mayRedo = useAppSelector((s) => s.regionalPrompts.future.length > 0);
  const redo = useCallback(() => {
    dispatch(redoRegionalPrompts());
  }, [dispatch]);
  useHotkeys(['meta+shift+z', 'ctrl+shift+z'], redo, { enabled: mayRedo, preventDefault: true }, [mayRedo, redo]);

  return (
    <ButtonGroup>
      <IconButton
        aria-label={t('unifiedCanvas.undo')}
        tooltip={t('unifiedCanvas.undo')}
        onClick={undo}
        icon={<PiArrowCounterClockwiseBold />}
        isDisabled={!mayUndo}
      />
      <IconButton
        aria-label={t('unifiedCanvas.redo')}
        tooltip={t('unifiedCanvas.redo')}
        onClick={redo}
        icon={<PiArrowClockwiseBold />}
        isDisabled={!mayRedo}
      />
    </ButtonGroup>
  );
});

UndoRedoButtonGroup.displayName = 'UndoRedoButtonGroup';
