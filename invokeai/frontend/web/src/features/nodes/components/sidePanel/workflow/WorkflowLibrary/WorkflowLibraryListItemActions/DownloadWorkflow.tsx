import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useDownloadWorkflow } from 'features/workflowLibrary/hooks/useDownloadWorkflow';
import type { MouseEvent } from 'react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDownloadSimpleBold } from 'react-icons/pi';

export const DownloadWorkflow = ({
  isHovered,
  setIsHovered,
}: {
  isHovered: boolean;
  setIsHovered: (isHovered: boolean) => void;
}) => {
  const downloadWorkflow = useDownloadWorkflow();
  const handleClickDownload = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      setIsHovered(false);
      downloadWorkflow();
    },
    [downloadWorkflow, setIsHovered]
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
      />
    </Tooltip>
  );
};
