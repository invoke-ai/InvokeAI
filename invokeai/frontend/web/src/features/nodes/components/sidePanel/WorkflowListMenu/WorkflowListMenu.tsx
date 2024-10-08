import { Flex, Spinner } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { EMPTY_ARRAY } from 'app/store/constants';
import { $projectId } from 'app/store/nanostores/projectId';
import { $workflowCategories } from 'app/store/nanostores/workflowCategories';
import { useAppSelector } from 'app/store/storeHooks';
import { selectWorkflowSearchTerm } from 'features/nodes/store/workflowSlice';
import UploadWorkflowButton from 'features/workflowLibrary/components/UploadWorkflowButton';
import { useTranslation } from 'react-i18next';
import { useListWorkflowsQuery } from 'services/api/endpoints/workflows';
import type { WorkflowRecordListItemDTO } from 'services/api/types';

import { WorkflowList } from './WorkflowList';
import WorkflowSearch from './WorkflowSearch';

export const WorkflowListMenu = () => {
  const searchTerm = useAppSelector(selectWorkflowSearchTerm);
  const projectId = useStore($projectId);
  const { data, isLoading } = useListWorkflowsQuery(
    { order_by: projectId ? 'created_at' : 'opened_at' },
    {
      selectFromResult: ({ data, isLoading }) => {
        const filteredData =
          data?.items.filter((workflow) => workflow.name.toLowerCase().includes(searchTerm.toLowerCase())) ||
          EMPTY_ARRAY;

        const groupedData = filteredData.reduce(
          (
            acc: {
              default: WorkflowRecordListItemDTO[];
              project: WorkflowRecordListItemDTO[];
              user: WorkflowRecordListItemDTO[];
            },
            workflow
          ) => {
            if (workflow.category === 'default') {
              acc.default.push(workflow);
            } else if (workflow.category === 'project') {
              acc.project.push(workflow);
            } else {
              acc.user.push(workflow);
            }
            return acc;
          },
          { default: [], project: [], user: [] }
        );

        return {
          data: groupedData,
          isLoading,
        };
      },
    }
  );

  const workflowCategories = useStore($workflowCategories);

  const { t } = useTranslation();

  return (
    <Flex flexDir="column" gap={2} padding={3} layerStyle="second" borderRadius="base">
      <Flex alignItems="center" gap={2} w="full" justifyContent="space-between">
        <WorkflowSearch />
        <UploadWorkflowButton />
      </Flex>

      {isLoading ? (
        <Flex alignItems="center" justifyContent="center" p={20}>
          <Spinner />
        </Flex>
      ) : (
        <>
          {workflowCategories.map((category) => (
            <WorkflowList key={category} title={t(`workflows.${category}Workflows`)} data={data[category]} />
          ))}
        </>
      )}
    </Flex>
  );
};
