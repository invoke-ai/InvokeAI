import { Box } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { overlayScrollbarsParams } from 'common/components/OverlayScrollbars/constants';
import { selectAllowPrivateBoards } from 'features/system/store/configSelectors';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import type { CSSProperties } from 'react';
import { memo } from 'react';

import { BoardsList } from './BoardsList';

const overlayScrollbarsStyles: CSSProperties = {
  height: '100%',
  width: '100%',
};

const BoardsListWrapper = () => {
  const allowPrivateBoards = useAppSelector(selectAllowPrivateBoards);

  return (
    <Box position="relative" w="full" h="full">
      <Box position="absolute" top={0} right={0} bottom={0} left={0}>
        <OverlayScrollbarsComponent defer style={overlayScrollbarsStyles} options={overlayScrollbarsParams.options}>
          {allowPrivateBoards && <BoardsList isPrivate={true} />}
          <BoardsList isPrivate={false} />
        </OverlayScrollbarsComponent>
      </Box>
    </Box>
  );
};
export default memo(BoardsListWrapper);
