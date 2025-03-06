import { Button } from '@invoke-ai/ui-library';
import { useWorkflowLibraryModal } from 'features/nodes/store/workflowLibraryModal';
import { saveWorkflowAs } from 'features/workflowLibrary/components/SaveWorkflowAsDialog';
import { useLoadWorkflowFromFile } from 'features/workflowLibrary/hooks/useLoadWorkflowFromFile';
import { memo, useCallback, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import { PiUploadSimpleBold } from 'react-icons/pi';

export const UploadWorkflowButton = memo(() => {
  const { t } = useTranslation();
  const resetRef = useRef<() => void>(null);
  const workflowLibraryModal = useWorkflowLibraryModal();

  const loadWorkflowFromFile = useLoadWorkflowFromFile({
    resetRef,
    onSuccess: (workflow) => {
      workflowLibraryModal.close();
      saveWorkflowAs(workflow);
    },
  });

  const onDropAccepted = useCallback(
    (files: File[]) => {
      if (!files[0]) {
        return;
      }
      loadWorkflowFromFile(files[0]);
    },
    [loadWorkflowFromFile]
  );

  const { getInputProps, getRootProps } = useDropzone({
    accept: { 'application/json': ['.json'] },
    onDropAccepted,
    noDrag: true,
    multiple: false,
  });
  return (
    <>
      <Button leftIcon={<PiUploadSimpleBold />} {...getRootProps()} pointerEvents="auto" variant="ghost">
        {t('workflows.uploadWorkflow')}
      </Button>

      <input {...getInputProps()} />
    </>
  );
});

UploadWorkflowButton.displayName = 'UploadWorkflowButton';
