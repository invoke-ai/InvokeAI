import { MenuItem } from '@invoke-ai/ui-library';
import { useDownloadWorkflow } from 'features/workflowLibrary/hooks/useDownloadWorkflow';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDownloadSimpleBold } from 'react-icons/pi';

const DownloadWorkflowMenuItem = () => {
  const { t } = useTranslation();
  const downloadWorkflow = useDownloadWorkflow();

  return (
    <MenuItem as="button" icon={<PiDownloadSimpleBold />} onClick={downloadWorkflow}>
      {t('workflows.downloadWorkflow')}
    </MenuItem>
  );
};

export default memo(DownloadWorkflowMenuItem);
