import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { autoScrollForElements } from '@atlaskit/pragmatic-drag-and-drop-auto-scroll/element';
import { autoScrollForExternal } from '@atlaskit/pragmatic-drag-and-drop-auto-scroll/external';
import { Box } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { overlayScrollbarsParams } from 'common/components/OverlayScrollbars/constants';
import { selectAllowPrivateBoards } from 'features/system/store/configSelectors';
import type { OverlayScrollbarsComponentRef } from 'overlayscrollbars-react';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import type { CSSProperties } from 'react';
import { memo, useEffect, useState } from 'react';

import { BoardsList } from './BoardsList';

const overlayScrollbarsStyles: CSSProperties = {
  height: '100%',
  width: '100%',
};

const BoardsListWrapper = () => {
  const allowPrivateBoards = useAppSelector(selectAllowPrivateBoards);
  const [os, osRef] = useState<OverlayScrollbarsComponentRef | null>(null);
  useEffect(() => {
    const element = os?.osInstance()?.elements().viewport;
    if (!element) {
      return;
    }
    return combine(autoScrollForElements({ element }), autoScrollForExternal({ element }));
  }, [os]);

  return (
    <Box position="relative" w="full" h="full">
      <Box position="absolute" top={0} right={0} bottom={0} left={0}>
        <OverlayScrollbarsComponent
          ref={osRef}
          style={overlayScrollbarsStyles}
          options={overlayScrollbarsParams.options}
        >
          {allowPrivateBoards && <BoardsList isPrivate={true} />}
          <BoardsList isPrivate={false} />
        </OverlayScrollbarsComponent>
      </Box>
    </Box>
  );
};
export default memo(BoardsListWrapper);
