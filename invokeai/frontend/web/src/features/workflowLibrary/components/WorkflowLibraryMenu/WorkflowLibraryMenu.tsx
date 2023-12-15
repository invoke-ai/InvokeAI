import {
  Menu,
  MenuButton,
  MenuDivider,
  MenuList,
  useDisclosure,
} from '@chakra-ui/react';
import IAIIconButton from 'common/components/IAIIconButton';
import { useGlobalMenuCloseTrigger } from 'common/hooks/useGlobalMenuCloseTrigger';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import DownloadWorkflowMenuItem from 'features/workflowLibrary/components/WorkflowLibraryMenu/DownloadWorkflowMenuItem';
import NewWorkflowMenuItem from 'features/workflowLibrary/components/WorkflowLibraryMenu/NewWorkflowMenuItem';
import SaveWorkflowAsMenuItem from 'features/workflowLibrary/components/WorkflowLibraryMenu/SaveWorkflowAsMenuItem';
import SaveWorkflowMenuItem from 'features/workflowLibrary/components/WorkflowLibraryMenu/SaveWorkflowMenuItem';
import SettingsMenuItem from 'features/workflowLibrary/components/WorkflowLibraryMenu/SettingsMenuItem';
import UploadWorkflowMenuItem from 'features/workflowLibrary/components/WorkflowLibraryMenu/UploadWorkflowMenuItem';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaEllipsis } from 'react-icons/fa6';
import { menuListMotionProps } from 'theme/components/menu';

const WorkflowLibraryMenu = () => {
  const { t } = useTranslation();
  const { isOpen, onOpen, onClose } = useDisclosure();
  useGlobalMenuCloseTrigger(onClose);
  const isWorkflowLibraryEnabled =
    useFeatureStatus('workflowLibrary').isFeatureEnabled;

  return (
    <Menu isOpen={isOpen} onOpen={onOpen} onClose={onClose}>
      <MenuButton
        as={IAIIconButton}
        aria-label={t('workflows.workflowEditorMenu')}
        icon={<FaEllipsis />}
        pointerEvents="auto"
      />
      <MenuList motionProps={menuListMotionProps} pointerEvents="auto">
        {isWorkflowLibraryEnabled && <SaveWorkflowMenuItem />}
        {isWorkflowLibraryEnabled && <SaveWorkflowAsMenuItem />}
        <DownloadWorkflowMenuItem />
        <UploadWorkflowMenuItem />
        <NewWorkflowMenuItem />
        <MenuDivider />
        <SettingsMenuItem />
      </MenuList>
    </Menu>
  );
};

export default memo(WorkflowLibraryMenu);
