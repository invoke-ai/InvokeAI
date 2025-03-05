import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useLoadWorkflow } from 'features/workflowLibrary/components/LoadWorkflowConfirmationAlertDialog';
import type { MouseEvent } from 'react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFloppyDiskBold } from 'react-icons/pi';

// needs to clone and save workflow to account without taking over editor
export const SaveWorkflow = ({ workflowId }: { workflowId: string }) => {
  const loadWorkflow = useLoadWorkflow();
  const { t } = useTranslation();

  const handleClickSave = useCallback(
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
        onClick={handleClickSave}
        icon={<PiFloppyDiskBold />}
      />
    </Tooltip>
  );
};
