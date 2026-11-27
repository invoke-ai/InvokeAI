import { MenuItem } from '@invoke-ai/ui-library';
import { useNewWorkflow } from 'features/workflowLibrary/components/NewWorkflowConfirmationAlertDialog';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFilePlusBold } from 'react-icons/pi';

export const NewWorkflowMenuItem = memo(() => {
  const { t } = useTranslation();
  const newWorkflow = useNewWorkflow();

  return (
    <MenuItem as="button" icon={<PiFilePlusBold />} onClick={newWorkflow.createWithDialog}>
      {t('nodes.newWorkflow')}
    </MenuItem>
  );
});

NewWorkflowMenuItem.displayName = 'NewWorkflowMenuItem';
