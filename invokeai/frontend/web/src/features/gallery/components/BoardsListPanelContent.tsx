import { Box, Collapse, Divider, Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { BoardsListWrapper } from 'features/gallery/components/Boards/BoardsList/BoardsListWrapper';
import { $boardSearchIsOpen, BoardsSearch } from 'features/gallery/components/Boards/BoardsList/BoardsSearch';
import { GalleryTopBar } from 'features/gallery/components/GalleryTopBar';
import type { CSSProperties } from 'react';
import { memo } from 'react';

const COLLAPSE_STYLES: CSSProperties = { flexShrink: 0, minHeight: 0 };

export const BoardsPanel = memo(() => {
  const boardSearchDisclosure = useStore($boardSearchIsOpen);
  return (
    <Flex flexDir="column" w="full" h="full" p={2}>
      <GalleryTopBar />
      <Collapse in={boardSearchDisclosure} style={COLLAPSE_STYLES}>
        <Box w="full" pt={2}>
          <BoardsSearch />
        </Box>
      </Collapse>
      <Divider pt={2} />
      <BoardsListWrapper />
    </Flex>
  );
});
BoardsPanel.displayName = 'BoardsPanel';
