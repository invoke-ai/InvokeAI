import IAIIconButton from 'common/components/IAIIconButton';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaUndo, FaRedo } from 'react-icons/fa';
import { undoAction, redoAction } from 'features/nodes/store/actions';

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

  return (
    <>
      <IAIIconButton
        icon={<FaUndo />}
        tooltip={t('nodes.undo')}
        aria-label={t('nodes.undo')}
        onClick={handleUndo}
        disabled={!canUndo}
      />
      <IAIIconButton
        icon={<FaRedo />}
        tooltip={t('nodes.redo')}
        aria-label={t('nodes.redo')}
        onClick={handleRedo}
        disabled={!canRedo}
      />
    </>
  );
};

export default memo(UndoRedoButton);
