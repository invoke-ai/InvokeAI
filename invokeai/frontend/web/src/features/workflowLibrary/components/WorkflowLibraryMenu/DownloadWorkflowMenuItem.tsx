import { InvMenuItem } from 'common/components/InvMenu/InvMenuItem';
import { useDownloadWorkflow } from 'features/workflowLibrary/hooks/useDownloadWorkflow';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaDownload } from 'react-icons/fa';

const DownloadWorkflowMenuItem = () => {
  const { t } = useTranslation();
  const downloadWorkflow = useDownloadWorkflow();

  return (
    <InvMenuItem as="button" icon={<FaDownload />} onClick={downloadWorkflow}>
      {t('workflows.downloadWorkflow')}
    </InvMenuItem>
  );
};

export default memo(DownloadWorkflowMenuItem);
