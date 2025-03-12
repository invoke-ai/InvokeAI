import { Button } from '@invoke-ai/ui-library';
import { useWorkflowLibraryModal } from 'features/nodes/store/workflowLibraryModal';
import { useLoadWorkflowFromFile } from 'features/workflowLibrary/hooks/useLoadWorkflowFromFile';
import { memo, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import { PiUploadSimpleBold } from 'react-icons/pi';

export const UploadWorkflowButton = memo(() => {
  const { t } = useTranslation();
  const workflowLibraryModal = useWorkflowLibraryModal();

  const loadWorkflowFromFile = useLoadWorkflowFromFile();

  const onDropAccepted = useCallback(
    ([file]: File[]) => {
      if (!file) {
        return;
      }
      loadWorkflowFromFile(file, {
        onSuccess: () => {
          workflowLibraryModal.close();
        },
      });
    },
    [loadWorkflowFromFile, workflowLibraryModal]
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
