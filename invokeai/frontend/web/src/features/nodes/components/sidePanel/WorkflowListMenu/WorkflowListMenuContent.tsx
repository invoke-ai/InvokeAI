import { Box, Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $workflowCategories } from 'app/store/nanostores/workflowCategories';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import UploadWorkflowButton from 'features/workflowLibrary/components/UploadWorkflowButton';
import type { RefObject } from 'react';
import { memo } from 'react';

import { WorkflowList } from './WorkflowList';
import { WorkflowSearch } from './WorkflowSearch';
import { WorkflowSortControl } from './WorkflowSortControl';

export const WorkflowListMenuContent = memo(({ searchInputRef }: { searchInputRef: RefObject<HTMLInputElement> }) => {
  const workflowCategories = useStore($workflowCategories);

  return (
    <Flex w="full" h="full" flexDir="column" gap={2}>
      <Flex alignItems="center" gap={2} w="full" justifyContent="space-between">
        <WorkflowSearch searchInputRef={searchInputRef} />
        <WorkflowSortControl />
        <UploadWorkflowButton />
      </Flex>

      <Box position="relative" w="full" h="full">
        <ScrollableContent>
          {workflowCategories.map((category) => (
            <WorkflowList key={category} category={category} />
          ))}
        </ScrollableContent>
      </Box>
    </Flex>
  );
});

WorkflowListMenuContent.displayName = 'WorkflowListMenuContent';
