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
    const osInstance = os?.osInstance();

    if (!osInstance) {
      return;
    }

    const element = osInstance.elements().viewport;

    // `pragmatic-drag-and-drop-auto-scroll` requires the element to have `overflow-y: scroll` or `overflow-y: auto`
    // else it logs an ugly warning. In our case, using a custom scrollbar library, it will be 'hidden' by default.
    // To prevent the erroneous warning, we temporarily set the overflow-y to 'scroll' and then revert it back.
    const overflowY = element.style.overflowY; // starts 'hidden'
    element.style.setProperty('overflow-y', 'scroll', 'important');
    const cleanup = combine(autoScrollForElements({ element }), autoScrollForExternal({ element }));
    element.style.setProperty('overflow-y', overflowY);

    return cleanup;
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
