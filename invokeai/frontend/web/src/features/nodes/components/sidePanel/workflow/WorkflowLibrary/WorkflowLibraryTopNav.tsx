import { Flex } from '@invoke-ai/ui-library';
import { useRef } from 'react';

import { WorkflowSearch } from './WorkflowSearch';
import { WorkflowSortControl } from './WorkflowSortControl';

export const WorkflowLibraryTopNav = () => {
  const searchInputRef = useRef<HTMLInputElement>(null);
  return (
    <Flex gap={8} justifyContent="space-between">
      <WorkflowSearch searchInputRef={searchInputRef} />
      <WorkflowSortControl />
    </Flex>
  );
};
