import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useLoadWorkflow } from 'features/workflowLibrary/components/LoadWorkflowConfirmationAlertDialog';
import type { MouseEvent } from 'react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPencilBold } from 'react-icons/pi';

export const EditWorkflow = ({ workflowId }: { workflowId: string }) => {
  const loadWorkflow = useLoadWorkflow();
  const { t } = useTranslation();

  const handleClickEdit = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      loadWorkflow.loadWithDialog(workflowId, 'edit');
    },
    [loadWorkflow, workflowId]
  );

  return (
    <Tooltip label={t('workflows.edit')} closeOnScroll>
      <IconButton
        size="sm"
        variant="ghost"
        aria-label={t('workflows.edit')}
        onClick={handleClickEdit}
        icon={<PiPencilBold />}
      />
    </Tooltip>
  );
};
