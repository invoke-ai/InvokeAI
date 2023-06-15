import { Box, Grid, Input, Spacer } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import {
  boardsSelector,
  selectBoardsAll,
  setBoardSearchText,
} from 'features/gallery/store/boardSlice';
import { memo, useEffect, useState } from 'react';
import HoverableBoard from './HoverableBoard';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import AddBoardButton from './AddBoardButton';
import AllImagesBoard from './AllImagesBoard';
import { searchBoardsSelector } from '../../store/boardSelectors';
import { useSelector } from 'react-redux';

const selector = createSelector(
  [selectBoardsAll, boardsSelector],
  (boards, boardsState) => {
    return { boards, selectedBoardId: boardsState.selectedBoardId };
  },
  defaultSelectorOptions
);

const BoardsList = () => {
  const dispatch = useAppDispatch();
  const { selectedBoardId } = useAppSelector(selector);
  const filteredBoards = useSelector(searchBoardsSelector);

  const [searchMode, setSearchMode] = useState(false);

  const handleBoardSearch = (searchTerm: string) => {
    setSearchMode(searchTerm.length > 0);
    dispatch(setBoardSearchText(searchTerm));
  };

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
      <Box margin="1rem 0">
        <Input
          placeholder="Search Boards..."
          onChange={(e) => {
            handleBoardSearch(e.target.value);
          }}
        />
      </Box>
      <Grid
        className="list-container"
        sx={{
          gap: 2,
          gridTemplateRows: '5rem 5rem',
          gridAutoFlow: 'column dense',
          gridAutoColumns: '4rem',
        }}
      >
        {!searchMode && (
          <>
            <AddBoardButton />
            <AllImagesBoard isSelected={selectedBoardId === null} />
          </>
        )}
        {filteredBoards.map((board) => (
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
