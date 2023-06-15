import { Grid } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { selectBoardsAll } from 'features/gallery/store/boardSlice';
import { memo } from 'react';
import HoverableBoard from './HoverableBoard';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import AddBoardButton from './AddBoardButton';

const selector = createSelector(
  selectBoardsAll,
  (boards) => {
    return { boards };
  },
  defaultSelectorOptions
);

const BoardsList = () => {
  const { boards } = useAppSelector(selector);

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
        {boards.map((board) => (
          <HoverableBoard key={board.board_id} board={board} />
        ))}
      </Grid>
    </OverlayScrollbarsComponent>
  );
};

export default memo(BoardsList);
