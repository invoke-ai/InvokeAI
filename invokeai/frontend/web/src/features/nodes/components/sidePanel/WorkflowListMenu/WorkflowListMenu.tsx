import { Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $workflowCategories } from 'app/store/nanostores/workflowCategories';
import UploadWorkflowButton from 'features/workflowLibrary/components/UploadWorkflowButton';

import { WorkflowList } from './WorkflowList';
import WorkflowSearch from './WorkflowSearch';
import { WorkflowSortControl } from './WorkflowSortControl';

export const WorkflowListMenu = () => {
  const workflowCategories = useStore($workflowCategories);

  return (
    <Flex flexDir="column" gap={2} padding={3} layerStyle="second" borderRadius="base">
      <Flex alignItems="center" gap={2} w="full" justifyContent="space-between">
        <WorkflowSearch />

        <WorkflowSortControl />

        <UploadWorkflowButton />
      </Flex>

      {workflowCategories.map((category) => (
        <WorkflowList key={category} category={category} />
      ))}
    </Flex>
  );
};
