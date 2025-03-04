import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $projectUrl } from 'app/store/nanostores/projectId';
import { useShareWorkflow } from 'features/nodes/components/sidePanel/workflow/WorkflowLibrary/ShareWorkflowModal';
import type { MouseEvent } from 'react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiShareFatBold } from 'react-icons/pi';
import type { WorkflowRecordListItemWithThumbnailDTO } from 'services/api/types';

export const ViewWorkflow = ({
  isHovered,
  setIsHovered,
  workflow,
}: {
  isHovered: boolean;
  setIsHovered: (isHovered: boolean) => void;
  workflow: WorkflowRecordListItemWithThumbnailDTO;
}) => {
  const projectUrl = useStore($projectUrl);
  const shareWorkflow = useShareWorkflow();
  const { t } = useTranslation();

  const handleClickShare = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      setIsHovered(false);
      shareWorkflow(workflow);
    },
    [shareWorkflow, workflow, setIsHovered]
  );

  if (!projectUrl || !workflow.workflow_id || workflow.category === 'user') {
    return null;
  }

  return (
    <Tooltip
      label={t('workflows.copyShareLink')}
      // This prevents an issue where the tooltip isn't closed after the modal is opened
      isOpen={!isHovered ? false : undefined}
      closeOnScroll
    >
      <IconButton
        size="sm"
        variant="ghost"
        aria-label={t('workflows.copyShareLink')}
        onClick={handleClickShare}
        icon={<PiShareFatBold />}
      />
    </Tooltip>
  );
};
