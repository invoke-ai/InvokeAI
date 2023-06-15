import { Grid } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import {
  boardsSelector,
  selectBoardsAll,
} from 'features/gallery/store/boardSlice';
import { memo, useState } from 'react';
import HoverableBoard from './HoverableBoard';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import AddBoardButton from './AddBoardButton';
import AllImagesBoard from './AllImagesBoard';

const selector = createSelector(
  [selectBoardsAll, boardsSelector],
  (boards, boardsState) => {
    return { boards, selectedBoardId: boardsState.selectedBoardId };
  },
  defaultSelectorOptions
);

const BoardsList = () => {
  const { boards, selectedBoardId } = useAppSelector(selector);

  return (
    <OverlayScrollbarsComponent
      defer
      style={{ height: '100%', width: '100%' }}
      options={{
        scrollbars: {
          visibility: 'auto',
          autoHide: 'move',
          autoHideDelay: 1300,
          theme: 'os-theme-dark',
        },
      }}
    >
      <Grid
        className="list-container"
        sx={{
          gap: 2,
          gridTemplateRows: '5rem 5rem',
          gridAutoFlow: 'column dense',
          gridAutoColumns: '4rem',
        }}
      >
        <AddBoardButton />
        <AllImagesBoard isSelected={selectedBoardId === null} />
        {boards.map((board) => (
          <HoverableBoard
            key={board.board_id}
            board={board}
            isSelected={selectedBoardId === board.board_id}
          />
        ))}
      </Grid>
    </OverlayScrollbarsComponent>
  );
};

export default memo(BoardsList);
