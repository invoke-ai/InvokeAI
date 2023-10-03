import IAIIconButton from 'common/components/IAIIconButton';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaUndo, FaRedo } from 'react-icons/fa';
import { undoAction, redoAction } from 'features/nodes/store/actions';
import { useHotkeys } from 'react-hotkeys-hook';
import { ButtonGroup } from '@chakra-ui/react';

const UndoRedoButton = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const canUndo = useAppSelector((state) => {
    return state.nodes.past.length > 0;
  });

  const canRedo = useAppSelector((state) => {
    return state.nodes.future.length > 0;
  });

  const handleUndo = () => {
    dispatch(undoAction());
  };

  const handleRedo = () => {
    dispatch(redoAction());
  };

  useHotkeys('meta+z, ctrl+z', handleUndo);
  useHotkeys('meta+shift+z, ctrl+shift+z', handleRedo);

  return (
    <ButtonGroup isAttached>
      <IAIIconButton
        icon={<FaUndo />}
        tooltip={`${t('nodes.undo')} (ctrl + Z)`}
        aria-label={`${t('nodes.undo')} (ctrl + Z)`}
        onClick={handleUndo}
        isDisabled={!canUndo}
      />
      <IAIIconButton
        icon={<FaRedo />}
        tooltip={`${t('nodes.redo')} (ctrl + shift + Z)`}
        aria-label={`${t('nodes.redo')} (ctrl + shift + Z)`}
        onClick={handleRedo}
        isDisabled={!canRedo}
      />
    </ButtonGroup>
  );
};

export default memo(UndoRedoButton);
