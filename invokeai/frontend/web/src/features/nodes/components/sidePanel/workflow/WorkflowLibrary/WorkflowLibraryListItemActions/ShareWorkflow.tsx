import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useShareWorkflow } from 'features/nodes/components/sidePanel/workflow/WorkflowLibrary/ShareWorkflowModal';
import type { MouseEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiShareFatBold } from 'react-icons/pi';
import type { WorkflowRecordListItemWithThumbnailDTO } from 'services/api/types';

export const ShareWorkflowButton = memo(({ workflow }: { workflow: WorkflowRecordListItemWithThumbnailDTO }) => {
  const shareWorkflow = useShareWorkflow();
  const { t } = useTranslation();

  const handleClickShare = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      shareWorkflow(workflow);
    },
    [shareWorkflow, workflow]
  );

  return (
    <Tooltip label={t('workflows.copyShareLink')} closeOnScroll>
      <IconButton
        size="sm"
        variant="link"
        alignSelf="stretch"
        aria-label={t('workflows.copyShareLink')}
        onClick={handleClickShare}
        icon={<PiShareFatBold />}
      />
    </Tooltip>
  );
});

ShareWorkflowButton.displayName = 'ShareWorkflowButton';
