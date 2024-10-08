import { Flex } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppSelector } from 'app/store/storeHooks';
import { selectStylePresetSearchTerm } from 'features/stylePresets/store/stylePresetSlice';
import { selectAllowPrivateStylePresets } from 'features/system/store/configSlice';
import UploadWorkflowButton from 'features/workflowLibrary/components/UploadWorkflowButton';
import { useTranslation } from 'react-i18next';
import { useListWorkflowsQuery } from 'services/api/endpoints/workflows';
import type { WorkflowRecordListItemDTO } from 'services/api/types';

import { WorkflowList } from './WorkflowList';
import WorkflowSearch from './WorkflowSearch';

export const WorkflowListMenu = () => {
  const searchTerm = useAppSelector(selectStylePresetSearchTerm);
  const allowProjectWorkflows = useAppSelector(selectAllowPrivateStylePresets);
  const { data } = useListWorkflowsQuery(
    {},
    {
      selectFromResult: ({ data }) => {
        const filteredData =
          data?.items.filter((workflow) => workflow.name.toLowerCase().includes(searchTerm.toLowerCase())) ||
          EMPTY_ARRAY;

        const groupedData = filteredData.reduce(
          (
            acc: {
              defaultWorkflows: WorkflowRecordListItemDTO[];
              sharedWorkflows: WorkflowRecordListItemDTO[];
              workflows: WorkflowRecordListItemDTO[];
            },
            workflow
          ) => {
            if (workflow.category === 'default') {
              acc.defaultWorkflows.push(workflow);
            } else if (workflow.category === 'project') {
              acc.sharedWorkflows.push(workflow);
            } else {
              acc.workflows.push(workflow);
            }
            return acc;
          },
          { defaultWorkflows: [], sharedWorkflows: [], workflows: [] }
        );

        return {
          data: groupedData,
        };
      },
    }
  );

  const { t } = useTranslation();

  return (
    <Flex flexDir="column" gap={2} padding={3} layerStyle="second" borderRadius="base">
      <Flex alignItems="center" gap={2} w="full" justifyContent="space-between">
        <WorkflowSearch />
        <UploadWorkflowButton />
      </Flex>

      <WorkflowList title="My Workflows" data={data.workflows} />
      {allowProjectWorkflows && <WorkflowList title={t('stylePresets.sharedTemplates')} data={data.sharedWorkflows} />}
      <WorkflowList title="Default Workflows" data={data.defaultWorkflows} />
    </Flex>
  );
};
