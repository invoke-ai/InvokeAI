import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useDownloadWorkflowById } from 'features/workflowLibrary/hooks/useDownloadWorkflowById';
import type { MouseEvent } from 'react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDownloadSimpleBold } from 'react-icons/pi';

// needs to be updated to work for a workflow other than the one loaded in editor
export const DownloadWorkflow = ({
  isHovered,
  setIsHovered,
  workflowId,
}: {
  isHovered: boolean;
  setIsHovered: (isHovered: boolean) => void;
  workflowId: string;
}) => {
  const downloadWorkflowById = useDownloadWorkflowById();
  const handleClickDownload = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      setIsHovered(false);
      downloadWorkflowById.downloadWorkflow(workflowId);
    },
    [downloadWorkflowById, setIsHovered, workflowId]
  );

  const { t } = useTranslation();
  return (
    <Tooltip
      label={t('workflows.download')}
      // This prevents an issue where the tooltip isn't closed after the modal is opened
      isOpen={!isHovered ? false : undefined}
      closeOnScroll
    >
      <IconButton
        size="sm"
        variant="ghost"
        aria-label={t('workflows.download')}
        onClick={handleClickDownload}
        icon={<PiDownloadSimpleBold />}
        isLoading={downloadWorkflowById.isLoading}
      />
    </Tooltip>
  );
};
