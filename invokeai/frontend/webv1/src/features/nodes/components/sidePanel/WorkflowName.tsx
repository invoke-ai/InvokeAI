import { Flex, Icon, Text, Tooltip } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useDoesWorkflowHaveUnsavedChanges } from 'features/nodes/components/sidePanel/workflow/IsolatedWorkflowBuilderWatcher';
import { selectWorkflowName } from 'features/nodes/store/selectors';
import { selectWorkflowMode } from 'features/nodes/store/workflowLibrarySlice';
import { useTranslation } from 'react-i18next';
import { PiDotOutlineFill } from 'react-icons/pi';

import WorkflowInfoTooltipContent from './viewMode/WorkflowInfoTooltipContent';
import { WorkflowWarning } from './viewMode/WorkflowWarning';

export const WorkflowName = () => {
  const { t } = useTranslation();
  const name = useAppSelector(selectWorkflowName);
  const doesWorkflowHaveUnsavedChanges = useDoesWorkflowHaveUnsavedChanges();
  const mode = useAppSelector(selectWorkflowMode);

  return (
    <Flex gap="1" alignItems="center">
      {name.length ? (
        <Tooltip label={<WorkflowInfoTooltipContent />} placement="top">
          <Text fontSize="lg" userSelect="none" noOfLines={1} wordBreak="break-all" fontWeight="semibold">
            {name}
          </Text>
        </Tooltip>
      ) : (
        <Text fontSize="lg" fontStyle="italic" fontWeight="semibold">
          {t('workflows.unnamedWorkflow')}
        </Text>
      )}

      {doesWorkflowHaveUnsavedChanges && mode === 'edit' && (
        <Tooltip label={t('nodes.newWorkflowDesc2')}>
          <Flex>
            <Icon as={PiDotOutlineFill} boxSize="20px" color="invokeYellow.500" />
          </Flex>
        </Tooltip>
      )}
      <WorkflowWarning />
    </Flex>
  );
};
