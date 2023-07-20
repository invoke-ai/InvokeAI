import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIButton from 'common/components/IAIButton';
import { boardIdSelected } from 'features/gallery/store/gallerySlice';
import { memo, useCallback, useMemo } from 'react';
import { useBoardName } from 'services/api/hooks/useBoardName';

type Props = {
  board_id: 'images' | 'assets' | 'no_board';
};

const SystemBoardButton = ({ board_id }: Props) => {
  const dispatch = useAppDispatch();

  const selector = useMemo(
    () =>
      createSelector(
        [stateSelector],
        ({ gallery }) => {
          const { selectedBoardId } = gallery;
          return { isSelected: selectedBoardId === board_id };
        },
        defaultSelectorOptions
      ),
    [board_id]
  );

  const { isSelected } = useAppSelector(selector);

  const boardName = useBoardName(board_id);

  const handleClick = useCallback(() => {
    dispatch(boardIdSelected(board_id));
  }, [board_id, dispatch]);

  return (
    <IAIButton
      onClick={handleClick}
      size="sm"
      isChecked={isSelected}
      sx={{
        flexGrow: 1,
        borderRadius: 'base',
      }}
    >
      {boardName}
    </IAIButton>
  );
};

export default memo(SystemBoardButton);
