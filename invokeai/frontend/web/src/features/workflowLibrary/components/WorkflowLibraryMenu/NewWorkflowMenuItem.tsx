import { MenuItem } from '@invoke-ai/ui-library';
import { NewWorkflowConfirmationAlertDialog } from 'features/workflowLibrary/components/NewWorkflowConfirmationAlertDialog';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFilePlusBold } from 'react-icons/pi';

export const NewWorkflowMenuItem = memo(() => {
  const { t } = useTranslation();

  const renderButton = useCallback(
    (onPointerUp: () => void) => (
      <MenuItem as="button" icon={<PiFilePlusBold />} onPointerUp={onPointerUp}>
        {t('nodes.newWorkflow')}
      </MenuItem>
    ),
    [t]
  );

  return <NewWorkflowConfirmationAlertDialog renderButton={renderButton} />;
});

NewWorkflowMenuItem.displayName = 'NewWorkflowMenuItem';
