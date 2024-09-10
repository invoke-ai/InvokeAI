import { Box } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { overlayScrollbarsParams } from 'common/components/OverlayScrollbars/constants';
import DeleteBoardModal from 'features/gallery/components/Boards/DeleteBoardModal';
import { selectAllowPrivateBoards } from 'features/system/store/configSelectors';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import type { CSSProperties } from 'react';
import { memo, useState } from 'react';
import type { BoardDTO } from 'services/api/types';

import { BoardsList } from './BoardsList';

const overlayScrollbarsStyles: CSSProperties = {
  height: '100%',
  width: '100%',
};

const BoardsListWrapper = () => {
  const allowPrivateBoards = useAppSelector(selectAllowPrivateBoards);
  const [boardToDelete, setBoardToDelete] = useState<BoardDTO>();

  return (
    <>
      <Box position="relative" w="full" h="full">
        <Box position="absolute" top={0} right={0} bottom={0} left={0}>
          <OverlayScrollbarsComponent defer style={overlayScrollbarsStyles} options={overlayScrollbarsParams.options}>
            {allowPrivateBoards && <BoardsList isPrivate={true} setBoardToDelete={setBoardToDelete} />}
            <BoardsList isPrivate={false} setBoardToDelete={setBoardToDelete} />
          </OverlayScrollbarsComponent>
        </Box>
      </Box>
      <DeleteBoardModal boardToDelete={boardToDelete} setBoardToDelete={setBoardToDelete} />
    </>
  );
};
export default memo(BoardsListWrapper);
