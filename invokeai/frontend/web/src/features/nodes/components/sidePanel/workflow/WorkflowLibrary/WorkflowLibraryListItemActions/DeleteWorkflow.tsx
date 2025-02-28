import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useDeleteWorkflow } from 'features/workflowLibrary/components/DeleteLibraryWorkflowConfirmationAlertDialog';
import type { MouseEvent } from 'react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashBold } from 'react-icons/pi';

export const DeleteWorkflow = ({
  isHovered,
  setIsHovered,
  workflowId,
}: {
  isHovered: boolean;
  setIsHovered: (isHovered: boolean) => void;
  workflowId: string;
}) => {
  const { t } = useTranslation();
  const deleteWorkflow = useDeleteWorkflow();

  const handleClickDelete = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      setIsHovered(false);
      deleteWorkflow(workflowId);
    },
    [deleteWorkflow, workflowId, setIsHovered]
  );
  return (
    <Tooltip
      label={t('workflows.delete')}
      // This prevents an issue where the tooltip isn't closed after the modal is opened
      isOpen={!isHovered ? false : undefined}
      closeOnScroll
    >
      <IconButton
        size="sm"
        variant="ghost"
        aria-label={t('workflows.delete')}
        onClick={handleClickDelete}
        colorScheme="error"
        icon={<PiTrashBold />}
      />
    </Tooltip>
  );
};
