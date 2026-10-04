import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useDeleteWorkflow } from 'features/workflowLibrary/components/DeleteLibraryWorkflowConfirmationAlertDialog';
import type { MouseEvent } from 'react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashBold } from 'react-icons/pi';

export const DeleteWorkflow = ({ workflowId }: { workflowId: string }) => {
  const { t } = useTranslation();
  const deleteWorkflow = useDeleteWorkflow();

  const handleClickDelete = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      deleteWorkflow(workflowId);
    },
    [deleteWorkflow, workflowId]
  );
  return (
    <Tooltip label={t('workflows.delete')} closeOnScroll>
      <IconButton
        size="sm"
        variant="link"
        alignSelf="stretch"
        aria-label={t('workflows.delete')}
        onClick={handleClickDelete}
        colorScheme="error"
        icon={<PiTrashBold />}
      />
    </Tooltip>
  );
};
