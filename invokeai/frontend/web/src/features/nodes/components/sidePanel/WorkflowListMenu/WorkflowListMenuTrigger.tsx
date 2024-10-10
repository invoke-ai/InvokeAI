import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, IconButton } from '@invoke-ai/ui-library';
import { useWorkflowListMenu } from 'features/nodes/store/workflowListMenu';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';

import { ActiveWorkflow } from './ActiveWorkflow';

const _hover: SystemStyleObject = {
  bg: 'base.750',
};

export const WorkflowListMenuTrigger = () => {
  const workflowListMenu = useWorkflowListMenu();
  const { t } = useTranslation();

  return (
    <Flex
      onClick={workflowListMenu.toggle}
      backgroundColor="base.800"
      justifyContent="space-between"
      alignItems="center"
      py={2}
      px={3}
      borderRadius="base"
      gap={2}
      role="button"
      _hover={_hover}
      transitionProperty="background-color"
      transitionDuration="normal"
      w="full"
      cursor="pointer"
    >
      <ActiveWorkflow />
      <IconButton aria-label={t('stylePresets.viewList')} variant="ghost" icon={<PiCaretDownBold />} size="sm" />
    </Flex>
  );
};
