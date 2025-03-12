import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useLoadWorkflow } from 'features/workflowLibrary/components/LoadWorkflowConfirmationAlertDialog';
import type { MouseEvent } from 'react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiEyeBold } from 'react-icons/pi';

export const ViewWorkflow = ({ workflowId }: { workflowId: string }) => {
  const loadWorkflow = useLoadWorkflow();
  const { t } = useTranslation();

  const handleClickLoad = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      loadWorkflow.loadWithDialog({ type: 'library', workflowId, mode: 'view' });
    },
    [loadWorkflow, workflowId]
  );

  return (
    <Tooltip label={t('workflows.view')} closeOnScroll>
      <IconButton
        size="sm"
        variant="ghost"
        aria-label={t('workflows.view')}
        onClick={handleClickLoad}
        icon={<PiEyeBold />}
      />
    </Tooltip>
  );
};
