import { Button } from '@invoke-ai/ui-library';
import { useWorkflowLibraryModalContext } from 'features/workflowLibrary/context/useWorkflowLibraryModalContext';
import { useLoadWorkflowFromFile } from 'features/workflowLibrary/hooks/useLoadWorkflowFromFile';
import { memo, useCallback, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import { PiUploadSimpleBold } from 'react-icons/pi';

const UploadWorkflowButton = () => {
  const { t } = useTranslation();
  const resetRef = useRef<() => void>(null);
  const { onClose } = useWorkflowLibraryModalContext();
  const loadWorkflowFromFile = useLoadWorkflowFromFile({ resetRef, onSuccess: onClose });

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
      <Button
        aria-label={t('workflows.uploadWorkflow')}
        tooltip={t('workflows.uploadWorkflow')}
        leftIcon={<PiUploadSimpleBold />}
        {...getRootProps()}
        pointerEvents="auto"
      >
        {t('workflows.uploadWorkflow')}
      </Button>
      <input {...getInputProps()} />
    </>
  );
};

export default memo(UploadWorkflowButton);
