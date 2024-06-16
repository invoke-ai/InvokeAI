/* eslint-disable i18next/no-literal-string */
import { ButtonGroup, IconButton } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiArrowClockwiseBold, PiArrowCounterClockwiseBold } from 'react-icons/pi';

export const UndoRedoButtonGroup = memo(() => {
  const { t } = useTranslation();

  const mayUndo = useAppSelector(() => false);
  const handleUndo = useCallback(() => {
    // TODO(psyche): Implement undo
    // dispatch(undo());
  }, []);
  useHotkeys(['meta+z', 'ctrl+z'], handleUndo, { enabled: mayUndo, preventDefault: true }, [mayUndo, handleUndo]);

  const mayRedo = useAppSelector(() => false);
  const handleRedo = useCallback(() => {
    // TODO(psyche): Implement redo
    // dispatch(redo());
  }, []);
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
