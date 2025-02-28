import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useLoadWorkflow } from 'features/workflowLibrary/components/LoadWorkflowConfirmationAlertDialog';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiEyeBold, PiFloppyDiskBold } from 'react-icons/pi';


// needs to clone and save workflow to account without taking over editor
export const SaveWorkflow = ({
  isHovered,
  setIsHovered,
  workflowId,
}: {
  isHovered: boolean;
  setIsHovered: (isHovered: boolean) => void;
  workflowId: string;
}) => {
  const loadWorkflow = useLoadWorkflow();
  const { t } = useTranslation();

  const handleClickSave = useCallback(() => {
    setIsHovered(false);
    loadWorkflow.loadWithDialog(workflowId, 'view');
  }, [loadWorkflow, workflowId, setIsHovered]);

  return (
    <Tooltip
      label={t('workflows.edit')}
      // This prevents an issue where the tooltip isn't closed after the modal is opened
      isOpen={!isHovered ? false : undefined}
      closeOnScroll
    >
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
