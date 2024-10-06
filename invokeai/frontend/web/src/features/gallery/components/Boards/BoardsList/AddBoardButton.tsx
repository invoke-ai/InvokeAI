import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { boardIdSelected, boardSearchTextChanged } from 'features/gallery/store/gallerySlice';
import { selectAllowPrivateBoards } from 'features/system/store/configSelectors';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';
import { useCreateBoardMutation } from 'services/api/endpoints/boards';

type Props = {
  isPrivateBoard: boolean;
};

const AddBoardButton = ({ isPrivateBoard }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const allowPrivateBoards = useAppSelector(selectAllowPrivateBoards);
  const [createBoard, { isLoading }] = useCreateBoardMutation();
  const label = useMemo(() => {
    if (!allowPrivateBoards) {
      return t('boards.addBoard');
    }
    if (isPrivateBoard) {
      return t('boards.addPrivateBoard');
    }
    return t('boards.addSharedBoard');
  }, [allowPrivateBoards, isPrivateBoard, t]);

  const handleCreateBoard = useCallback(async () => {
    try {
      const board = await createBoard({ board_name: t('boards.myBoard'), is_private: isPrivateBoard }).unwrap();
      dispatch(boardIdSelected({ boardId: board.board_id }));
      dispatch(boardSearchTextChanged(''));
    } catch {
      //no-op
    }
  }, [t, createBoard, isPrivateBoard, dispatch]);

  return (
    <IconButton
      icon={<PiPlusBold />}
      isLoading={isLoading}
      tooltip={label}
      aria-label={label}
      onClick={handleCreateBoard}
      size="md"
      data-testid="add-board-button"
      variant="link"
      w={8}
      h={8}
    />
  );
};

export default memo(AddBoardButton);
