import { useAppDispatch } from 'app/store/storeHooks';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import { boardCreated } from 'features/gallery/store/actions';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';
import { useCreateBoardMutation } from 'services/api/endpoints/boards';

const AddBoardButton = () => {
  const { t } = useTranslation();
  const [createBoard, { isLoading }] = useCreateBoardMutation();
  const dispatch = useAppDispatch();

  const DEFAULT_BOARD_NAME = t('boards.myBoard');
  const handleCreateBoard = useCallback(async () => {
    await createBoard(DEFAULT_BOARD_NAME).unwrap();
    dispatch(boardCreated());
  }, [createBoard, DEFAULT_BOARD_NAME, dispatch]);

  return (
    <InvIconButton
      icon={<PiPlusBold />}
      isLoading={isLoading}
      tooltip={t('boards.addBoard')}
      aria-label={t('boards.addBoard')}
      onClick={handleCreateBoard}
      size="sm"
      data-testid="add-board-button"
    />
  );
};

export default memo(AddBoardButton);
