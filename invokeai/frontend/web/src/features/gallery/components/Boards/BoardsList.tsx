import {
  Collapse,
  Flex,
  Grid,
  IconButton,
  Input,
  InputGroup,
  InputRightElement,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import {
  boardsSelector,
  setBoardSearchText,
} from 'features/gallery/store/boardSlice';
import { memo, useState } from 'react';
import HoverableBoard from './HoverableBoard';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import AddBoardButton from './AddBoardButton';
import AllImagesBoard from './AllImagesBoard';
import { CloseIcon } from '@chakra-ui/icons';
import { useListAllBoardsQuery } from 'services/apiSlice';

const selector = createSelector(
  [boardsSelector],
  (boardsState) => {
    const { selectedBoardId, searchText } = boardsState;
    return { selectedBoardId, searchText };
  },
  defaultSelectorOptions
);

type Props = {
  isOpen: boolean;
};

const BoardsList = (props: Props) => {
  const { isOpen } = props;
  const dispatch = useAppDispatch();
  const { selectedBoardId, searchText } = useAppSelector(selector);

  const { data: boards } = useListAllBoardsQuery();

  const filteredBoards = searchText
    ? boards?.filter((board) =>
        board.board_name.toLowerCase().includes(searchText.toLowerCase())
      )
    : boards;

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
    <Collapse in={isOpen} animateOpacity>
      <Flex
        sx={{
          flexDir: 'column',
          gap: 2,
          bg: 'base.800',
          borderRadius: 'base',
          p: 2,
          mt: 2,
        }}
      >
        <Flex sx={{ gap: 2, alignItems: 'center' }}>
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
                <IconButton
                  onClick={clearBoardSearch}
                  size="xs"
                  variant="ghost"
                  aria-label="Clear Search"
                  icon={<CloseIcon boxSize={3} />}
                />
              </InputRightElement>
            )}
          </InputGroup>
          <AddBoardButton />
        </Flex>
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
              gridTemplateRows: '5.5rem 5.5rem',
              gridAutoFlow: 'column dense',
              gridAutoColumns: '4rem',
            }}
          >
            {!searchMode && <AllImagesBoard isSelected={!selectedBoardId} />}
            {filteredBoards &&
              filteredBoards.map((board) => (
                <HoverableBoard
                  key={board.board_id}
                  board={board}
                  isSelected={selectedBoardId === board.board_id}
                />
              ))}
          </Grid>
        </OverlayScrollbarsComponent>
      </Flex>
    </Collapse>
  );
};

export default memo(BoardsList);
