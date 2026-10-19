import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useDownloadWorkflowById } from 'features/workflowLibrary/hooks/useDownloadWorkflowById';
import type { MouseEvent } from 'react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDownloadSimpleBold } from 'react-icons/pi';

export const DownloadWorkflow = ({ workflowId }: { workflowId: string }) => {
  const downloadWorkflowById = useDownloadWorkflowById();

  const handleClickDownload = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      downloadWorkflowById.downloadWorkflow(workflowId);
    },
    [downloadWorkflowById, workflowId]
  );

  const { t } = useTranslation();
  return (
    <Tooltip label={t('workflows.download')} closeOnScroll>
      <IconButton
        size="sm"
        variant="link"
        alignSelf="stretch"
        aria-label={t('workflows.download')}
        onClick={handleClickDownload}
        icon={<PiDownloadSimpleBold />}
        isLoading={downloadWorkflowById.isLoading}
      />
    </Tooltip>
  );
};
