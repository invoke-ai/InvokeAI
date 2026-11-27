import { Button } from '@invoke-ai/ui-library';
import { useLoadWorkflowWithDialog } from 'features/workflowLibrary/components/LoadWorkflowConfirmationAlertDialog';
import { memo, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import { PiUploadSimpleBold } from 'react-icons/pi';

export const UploadWorkflowButton = memo(() => {
  const { t } = useTranslation();
  const loadWorkflowWithDialog = useLoadWorkflowWithDialog();

  const onDropAccepted = useCallback(
    ([file]: File[]) => {
      if (!file) {
        return;
      }
      loadWorkflowWithDialog({
        type: 'file',
        data: file,
      });
    },
    [loadWorkflowWithDialog]
  );

  const { getInputProps, getRootProps } = useDropzone({
    accept: { 'application/json': ['.json'] },
    onDropAccepted,
    noDrag: true,
    multiple: false,
  });

  return (
    <>
      <Button
        leftIcon={<PiUploadSimpleBold />}
        {...getRootProps()}
        pointerEvents="auto"
        variant="ghost"
        justifyContent="flex-start"
      >
        {t('workflows.uploadWorkflow')}
      </Button>

      <input {...getInputProps()} />
    </>
  );
});

UploadWorkflowButton.displayName = 'UploadWorkflowButton';
