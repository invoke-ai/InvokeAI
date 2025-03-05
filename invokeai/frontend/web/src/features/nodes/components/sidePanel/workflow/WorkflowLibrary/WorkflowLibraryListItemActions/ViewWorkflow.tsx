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
      loadWorkflow.loadWithDialog(workflowId, 'view');
    },
    [loadWorkflow, workflowId]
  );

  return (
    <Tooltip label={t('workflows.edit')} closeOnScroll>
      <IconButton
        size="sm"
        variant="ghost"
        aria-label={t('workflows.edit')}
        onClick={handleClickLoad}
        icon={<PiEyeBold />}
      />
    </Tooltip>
  );
};
