import {
  Box,
  Divider,
  Grid,
  Input,
  InputGroup,
  InputRightElement,
  Spacer,
  useDisclosure,
} from '@chakra-ui/react';
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
import IAICollapse from '../../../../common/components/IAICollapse';
import { CloseIcon } from '@chakra-ui/icons';
import { useListBoardsQuery } from 'services/apiSlice';

const selector = createSelector(
  [selectBoardsAll, boardsSelector],
  (boards, boardsState) => {
    const selectedBoard = boards.find(
      (board) => board.board_id === boardsState.selectedBoardId
    );
    return { selectedBoard, searchText: boardsState.searchText };
  },
  defaultSelectorOptions
);

const BoardsList = () => {
  const dispatch = useAppDispatch();
  const { selectedBoard, searchText } = useAppSelector(selector);
  // const filteredBoards = useSelector(searchBoardsSelector);
  const { isOpen, onToggle } = useDisclosure();

  const { data } = useListBoardsQuery({ offset: 0, limit: 8 });

  const filteredBoards = searchText
    ? data?.items.filter((board) =>
        board.board_name.toLowerCase().includes(searchText.toLowerCase())
      )
    : data.items;

  const [searchMode, setSearchMode] = useState(false);

  const handleBoardSearch = (searchTerm: string) => {
    setSearchMode(searchTerm.length > 0);
    dispatch(setBoardSearchText(searchTerm));
  };
  const clearBoardSearch = () => {
    setSearchMode(false);
    dispatch(setBoardSearchText(''));
  };

  return (
    <IAICollapse label="Select Board" isOpen={isOpen} onToggle={onToggle}>
      <>
        <Box marginBottom="1rem">
          <InputGroup>
            <Input
              placeholder="Search Boards..."
              value={searchText}
              onChange={(e) => {
                handleBoardSearch(e.target.value);
              }}
            />
            {searchText && searchText.length && (
              <InputRightElement>
                <CloseIcon onClick={clearBoardSearch} cursor="pointer" />
              </InputRightElement>
            )}
          </InputGroup>
        </Box>
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
            {!searchMode && (
              <>
                <AddBoardButton />
                <AllImagesBoard isSelected={!selectedBoard} />
              </>
            )}
            {filteredBoards &&
              filteredBoards.map((board) => (
                <HoverableBoard
                  key={board.board_id}
                  board={board}
                  isSelected={selectedBoard?.board_id === board.board_id}
                />
              ))}
          </Grid>
        </OverlayScrollbarsComponent>
      </>
    </IAICollapse>
  );
};

export default memo(BoardsList);
