import { IconButton, Popover, PopoverBody, PopoverContent, PopoverTrigger, Portal } from '@invoke-ai/ui-library';
import { WorkflowListMenuContent } from 'features/nodes/components/sidePanel/WorkflowListMenu/WorkflowListMenuContent';
import { useWorkflowListMenu } from 'features/nodes/store/workflowListMenu';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';

export const WorkflowListMenuTrigger = () => {
  const workflowListMenu = useWorkflowListMenu();
  const { t } = useTranslation();

  return (
    <Popover isOpen={workflowListMenu.isOpen} onClose={workflowListMenu.close} onOpen={workflowListMenu.open}>
      <PopoverTrigger>
        <IconButton aria-label={t('stylePresets.viewList')} variant="ghost" icon={<PiCaretDownBold />} size="sm" />
      </PopoverTrigger>
      <Portal appendToParentPortal={false}>
        <PopoverContent p={4} w={512} maxW="full" minH={512} maxH="full">
          <PopoverBody flex="1 1 0">
            <WorkflowListMenuContent />
          </PopoverBody>
        </PopoverContent>
      </Portal>
    </Popover>
  );
};
