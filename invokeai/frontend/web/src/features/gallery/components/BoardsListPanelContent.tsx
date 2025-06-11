import { Flex, useDisclosure } from '@invoke-ai/ui-library';
import { BoardsListWrapper } from 'features/gallery/components/Boards/BoardsList/BoardsListWrapper';
import type { CSSProperties } from 'react';
import { memo } from 'react';

const COLLAPSE_STYLES: CSSProperties = { flexShrink: 0, minHeight: 0 };

export const BoardsListPanelContent = memo(() => {
  const boardSearchDisclosure = useDisclosure({ defaultIsOpen: false });
  return (
    <Flex flexDir="column" w="full" h="full" p={2}>
      {/* <Collapse in={boardSearchDisclosure.isOpen} style={COLLAPSE_STYLES}>
        <Box w="full" pt={2}>
          <BoardsSearch />
        </Box>
      </Collapse> */}
      {/* <Divider pt={2} /> */}
      <BoardsListWrapper />
    </Flex>
  );
});
BoardsListPanelContent.displayName = 'BoardsListPanelContent';
