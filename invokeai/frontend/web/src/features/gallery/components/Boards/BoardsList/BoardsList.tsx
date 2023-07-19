import {
  Collapse,
  Flex,
  Grid,
  GridItem,
  useDisclosure,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import { memo, useState } from 'react';
import { useListAllBoardsQuery } from 'services/api/endpoints/boards';
import { useFeatureStatus } from '../../../../system/hooks/useFeatureStatus';
import AddBoardButton from './AddBoardButton';
import AllAssetsBoard from './AllAssetsBoard';
import AllImagesBoard from './AllImagesBoard';
import BatchBoard from './BatchBoard';
import BoardsSearch from './BoardsSearch';
import GalleryBoard from './GalleryBoard';
import NoBoardBoard from './NoBoardBoard';
import DeleteBoardModal from '../DeleteBoardModal';
import { BoardDTO } from 'services/api/types';

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
  const { selectedBoardId, searchText } = useAppSelector(selector);
  const { data: boards } = useListAllBoardsQuery();
  const isBatchEnabled = useFeatureStatus('batches').isFeatureEnabled;
  const filteredBoards = searchText
    ? boards?.filter((board) =>
        board.board_name.toLowerCase().includes(searchText.toLowerCase())
      )
    : boards;
  const [boardToDelete, setBoardToDelete] = useState<BoardDTO>();
  const [searchMode, setSearchMode] = useState(false);

  return (
    <>
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
            <BoardsSearch setSearchMode={setSearchMode} />
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
                    <AllImagesBoard isSelected={selectedBoardId === 'images'} />
                  </GridItem>
                  <GridItem sx={{ p: 1.5 }}>
                    <AllAssetsBoard isSelected={selectedBoardId === 'assets'} />
                  </GridItem>
                  <GridItem sx={{ p: 1.5 }}>
                    <NoBoardBoard isSelected={selectedBoardId === 'no_board'} />
                  </GridItem>
                  {isBatchEnabled && (
                    <GridItem sx={{ p: 1.5 }}>
                      <BatchBoard isSelected={selectedBoardId === 'batch'} />
                    </GridItem>
                  )}
                </>
              )}
              {filteredBoards &&
                filteredBoards.map((board) => (
                  <GridItem key={board.board_id} sx={{ p: 1.5 }}>
                    <GalleryBoard
                      board={board}
                      isSelected={selectedBoardId === board.board_id}
                      setBoardToDelete={setBoardToDelete}
                    />
                  </GridItem>
                ))}
            </Grid>
          </OverlayScrollbarsComponent>
        </Flex>
      </Collapse>
      <DeleteBoardModal
        boardToDelete={boardToDelete}
        setBoardToDelete={setBoardToDelete}
      />
    </>
  );
};

export default memo(BoardsList);
