import { MenuItem } from '@chakra-ui/react';
import { useDownloadWorkflow } from 'features/nodes/hooks/useDownloadWorkflow';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaDownload } from 'react-icons/fa';

const DownloadWorkflowMenuItem = () => {
  const { t } = useTranslation();
  const downloadWorkflow = useDownloadWorkflow();

  return (
    <MenuItem as="button" icon={<FaDownload />} onClick={downloadWorkflow}>
      {t('workflows.downloadWorkflow')}
    </MenuItem>
  );
};

export default memo(DownloadWorkflowMenuItem);
