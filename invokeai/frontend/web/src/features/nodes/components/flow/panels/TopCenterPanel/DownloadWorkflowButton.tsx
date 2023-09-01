import IAIIconButton from 'common/components/IAIIconButton';
import { useWorkflow } from 'features/nodes/hooks/useWorkflow';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { FaDownload } from 'react-icons/fa';

const DownloadWorkflowButton = () => {
  const { t } = useTranslation();
  const workflow = useWorkflow();
  const handleDownload = useCallback(() => {
    const blob = new Blob([JSON.stringify(workflow, null, 2)]);
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${workflow.name || 'My Workflow'}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  }, [workflow]);
  return (
    <IAIIconButton
      icon={<FaDownload />}
      tooltip={t('nodes.downloadWorkflow')}
      aria-label={t('nodes.downloadWorkflow')}
      onClick={handleDownload}
    />
  );
};

export default memo(DownloadWorkflowButton);
