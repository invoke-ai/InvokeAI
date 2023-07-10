import { CloseIcon } from '@chakra-ui/icons';
import {
  Collapse,
  Flex,
  Grid,
  GridItem,
  IconButton,
  Input,
  InputGroup,
  InputRightElement,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { setBoardSearchText } from 'features/gallery/store/boardSlice';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import { memo, useState } from 'react';
import { useListAllBoardsQuery } from 'services/api/endpoints/boards';
import AddBoardButton from './AddBoardButton';
import AllImagesBoard from './AllImagesBoard';
import BatchBoard from './BatchBoard';
import GalleryBoard from './GalleryBoard';

const selector = createSelector(
  [stateSelector],
  ({ boards, gallery }) => {
    const { searchText } = boards;
    const { selectedBoardId } = gallery;
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
        layerStyle={'first'}
        sx={{
          flexDir: 'column',
          gap: 2,
          p: 2,
          mt: 2,
          borderRadius: 'base',
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
              gridTemplateRows: '6.5rem 6.5rem',
              gridAutoFlow: 'column dense',
              gridAutoColumns: '5rem',
            }}
          >
            {!searchMode && (
              <>
                <GridItem sx={{ p: 1.5 }}>
                  <AllImagesBoard isSelected={!selectedBoardId} />
                </GridItem>
                <GridItem sx={{ p: 1.5 }}>
                  <BatchBoard isSelected={selectedBoardId === 'batch'} />
                </GridItem>
              </>
            )}
            {filteredBoards &&
              filteredBoards.map((board) => (
                <GridItem key={board.board_id} sx={{ p: 1.5 }}>
                  <GalleryBoard
                    board={board}
                    isSelected={selectedBoardId === board.board_id}
                  />
                </GridItem>
              ))}
          </Grid>
        </OverlayScrollbarsComponent>
      </Flex>
    </Collapse>
  );
};

export default memo(BoardsList);
