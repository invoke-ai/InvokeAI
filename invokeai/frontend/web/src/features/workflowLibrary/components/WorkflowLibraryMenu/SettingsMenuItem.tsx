import { MenuItem } from '@chakra-ui/react';
import WorkflowEditorSettings from 'features/nodes/components/flow/panels/TopRightPanel/WorkflowEditorSettings';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaCog } from 'react-icons/fa';

const DownloadWorkflowMenuItem = () => {
  const { t } = useTranslation();

  return (
    <WorkflowEditorSettings>
      {({ onOpen }) => (
        <MenuItem as="button" icon={<FaCog />} onClick={onOpen}>
          {t('nodes.workflowSettings')}
        </MenuItem>
      )}
    </WorkflowEditorSettings>
  );
};

export default memo(DownloadWorkflowMenuItem);
