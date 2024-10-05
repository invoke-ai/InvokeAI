import { IconButton } from '@invoke-ai/ui-library';
import { NewWorkflowConfirmationAlertDialog } from 'features/workflowLibrary/components/NewWorkflowConfirmationAlertDialog';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFilePlusBold } from 'react-icons/pi';

export const NewWorkflowButton = memo(() => {
  const { t } = useTranslation();

  const renderButton = useCallback(
    (onPointerUp: () => void) => (
      <IconButton
        aria-label={t('nodes.newWorkflow')}
        tooltip={t('nodes.newWorkflow')}
        icon={<PiFilePlusBold />}
        onPointerUp={onPointerUp}
        pointerEvents="auto"
      />
    ),
    [t]
  );

  return <NewWorkflowConfirmationAlertDialog renderButton={renderButton} />;
});

NewWorkflowButton.displayName = 'NewWorkflowButton';
