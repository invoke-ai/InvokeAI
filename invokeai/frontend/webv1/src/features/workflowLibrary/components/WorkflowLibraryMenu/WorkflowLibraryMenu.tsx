import {
  IconButton,
  Menu,
  MenuButton,
  MenuDivider,
  MenuList,
  useDisclosure,
  useGlobalMenuClose,
  useShiftModifier,
} from '@invoke-ai/ui-library';
import DownloadWorkflowMenuItem from 'features/workflowLibrary/components/WorkflowLibraryMenu/DownloadWorkflowMenuItem';
import LoadWorkflowFromGraphMenuItem from 'features/workflowLibrary/components/WorkflowLibraryMenu/LoadWorkflowFromGraphMenuItem';
import { NewWorkflowMenuItem } from 'features/workflowLibrary/components/WorkflowLibraryMenu/NewWorkflowMenuItem';
import SaveWorkflowAsMenuItem from 'features/workflowLibrary/components/WorkflowLibraryMenu/SaveWorkflowAsMenuItem';
import SaveWorkflowMenuItem from 'features/workflowLibrary/components/WorkflowLibraryMenu/SaveWorkflowMenuItem';
import UploadWorkflowMenuItem from 'features/workflowLibrary/components/WorkflowLibraryMenu/UploadWorkflowMenuItem';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDotsThreeOutlineFill } from 'react-icons/pi';

export const WorkflowLibraryMenu = memo(() => {
  const { t } = useTranslation();
  const { isOpen, onOpen, onClose } = useDisclosure();
  const shift = useShiftModifier();
  useGlobalMenuClose(onClose);
  return (
    <Menu isOpen={isOpen} onOpen={onOpen} onClose={onClose}>
      <MenuButton
        as={IconButton}
        aria-label={t('workflows.workflowEditorMenu')}
        icon={<PiDotsThreeOutlineFill />}
        pointerEvents="auto"
        size="sm"
        variant="ghost"
      />
      <MenuList pointerEvents="auto">
        <NewWorkflowMenuItem />
        <UploadWorkflowMenuItem />
        <MenuDivider />
        <SaveWorkflowMenuItem />
        <SaveWorkflowAsMenuItem />
        <DownloadWorkflowMenuItem />
        {shift && <MenuDivider />}
        {shift && <LoadWorkflowFromGraphMenuItem />}
      </MenuList>
    </Menu>
  );
});
WorkflowLibraryMenu.displayName = 'WorkflowLibraryMenu';
