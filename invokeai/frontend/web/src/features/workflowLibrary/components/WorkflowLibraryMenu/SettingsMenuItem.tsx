import { MenuItem } from '@invoke-ai/ui-library';
import WorkflowEditorSettings from 'features/nodes/components/flow/panels/TopRightPanel/WorkflowEditorSettings';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { RiSettings4Line } from 'react-icons/ri';

const DownloadWorkflowMenuItem = () => {
  const { t } = useTranslation();

  return (
    <WorkflowEditorSettings>
      {({ onOpen }) => (
        <MenuItem as="button" icon={<RiSettings4Line />} onClick={onOpen}>
          {t('nodes.workflowSettings')}
        </MenuItem>
      )}
    </WorkflowEditorSettings>
  );
};

export default memo(DownloadWorkflowMenuItem);
