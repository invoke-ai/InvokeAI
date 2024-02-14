import { Flex, Icon, Text, Tooltip } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useTranslation } from 'react-i18next';
import { PiDotOutlineFill } from 'react-icons/pi';

import WorkflowInfoTooltipContent from './viewMode/WorkflowInfoTooltipContent';
import { WorkflowWarning } from './viewMode/WorkflowWarning';

export const WorkflowName = () => {
  const { name, isTouched, mode } = useAppSelector((s) => s.workflow);
  const { t } = useTranslation();

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

      {isTouched && mode === 'edit' && (
        <Tooltip label="Workflow has unsaved changes">
          <Flex>
            <Icon as={PiDotOutlineFill} boxSize="20px" sx={{ color: 'invokeYellow.500' }} />
          </Flex>
        </Tooltip>
      )}
      <WorkflowWarning />
    </Flex>
  );
};
