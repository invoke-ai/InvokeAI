import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectWorkflowLibraryView } from 'features/nodes/store/workflowLibrarySlice';
import { useRef } from 'react';

import { WorkflowSearch } from './WorkflowSearch';
import { WorkflowSortControl } from './WorkflowSortControl';

export const WorkflowLibraryTopNav = () => {
  const searchInputRef = useRef<HTMLInputElement>(null);
  const view = useAppSelector(selectWorkflowLibraryView);
  return (
    <Flex gap={8} justifyContent="space-between">
      <WorkflowSearch searchInputRef={searchInputRef} />
      {view !== 'recent' && <WorkflowSortControl />}
    </Flex>
  );
};
