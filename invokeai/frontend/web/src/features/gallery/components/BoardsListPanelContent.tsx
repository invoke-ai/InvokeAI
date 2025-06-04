import type { UseDisclosureReturn } from '@invoke-ai/ui-library';
import { Box, Collapse, Divider, Flex } from '@invoke-ai/ui-library';
import { BoardsListWrapper } from 'features/gallery/components/Boards/BoardsList/BoardsListWrapper';
import { BoardsSearch } from 'features/gallery/components/Boards/BoardsList/BoardsSearch';
import type { CSSProperties } from 'react';
import { memo } from 'react';

const COLLAPSE_STYLES: CSSProperties = { flexShrink: 0, minHeight: 0 };

export const BoardsListPanelContent = memo(
  ({ boardSearchDisclosure }: { boardSearchDisclosure: UseDisclosureReturn }) => {
    return (
      <Flex flexDir="column" w="full" h="full">
        <Collapse in={boardSearchDisclosure.isOpen} style={COLLAPSE_STYLES}>
          <Box w="full" pt={2}>
            <BoardsSearch />
          </Box>
        </Collapse>
        <Divider pt={2} />
        <BoardsListWrapper />
      </Flex>
    );
  }
);
BoardsListPanelContent.displayName = 'BoardsListPanelContent';
