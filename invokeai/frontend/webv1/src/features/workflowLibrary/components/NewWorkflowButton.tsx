import { Button } from '@invoke-ai/ui-library';
import { useNewWorkflow } from 'features/workflowLibrary/components/NewWorkflowConfirmationAlertDialog';
import type { MouseEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFilePlusBold } from 'react-icons/pi';

export const NewWorkflowButton = memo(() => {
  const newWorkflow = useNewWorkflow();

  const { t } = useTranslation();

  const onClickNewWorkflow = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      // We need to stop the event from propagating to the parent element, else the click will open the list menu
      e.stopPropagation();
      newWorkflow.createWithDialog();
    },
    [newWorkflow]
  );

  return (
    <Button onClick={onClickNewWorkflow} variant="ghost" leftIcon={<PiFilePlusBold />} justifyContent="flex-start">
      {t('nodes.newWorkflow')}
    </Button>
  );
});

NewWorkflowButton.displayName = 'NewWorkflowButton';
