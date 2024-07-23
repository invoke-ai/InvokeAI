import { Box, Spacer } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { overlayScrollbarsParams } from 'common/components/OverlayScrollbars/constants';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import type { CSSProperties } from 'react';
import { memo } from 'react';

import { BoardsList } from './BoardsList';
import NoBoardBoard from './NoBoardBoard';

const overlayScrollbarsStyles: CSSProperties = {
  height: '100%',
  width: '100%',
};

const BoardsListWrapper = () => {
  const selectedBoardId = useAppSelector((s) => s.gallery.selectedBoardId);
  const allowPrivateBoards = useAppSelector((s) => s.config.allowPrivateBoards);

  return (
    <>
      <Box position="relative" w="full" h="full">
        <Box position="absolute" top={0} right={0} bottom={0} left={0}>
          <OverlayScrollbarsComponent defer style={overlayScrollbarsStyles} options={overlayScrollbarsParams.options}>
            <Spacer pt="5px" />
            <NoBoardBoard isSelected={selectedBoardId === 'none'} />
            {allowPrivateBoards && <BoardsList isPrivate={true} />}
            <BoardsList />
          </OverlayScrollbarsComponent>
        </Box>
      </Box>
    </>
  );
};
export default memo(BoardsListWrapper);
