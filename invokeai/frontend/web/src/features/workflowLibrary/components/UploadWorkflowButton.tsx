import { IconButton } from '@invoke-ai/ui-library';
import { $isWorkflowListMenuIsOpen } from 'features/nodes/store/workflowListMenu';
import { useLoadWorkflowFromFile } from 'features/workflowLibrary/hooks/useLoadWorkflowFromFile';
import { memo, useCallback, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import { PiUploadSimpleBold } from 'react-icons/pi';

const UploadWorkflowButton = () => {
  const { t } = useTranslation();
  const resetRef = useRef<() => void>(null);

  const loadWorkflowFromFile = useLoadWorkflowFromFile({
    resetRef,
    onSuccess: () => $isWorkflowListMenuIsOpen.set(false),
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
      <IconButton
        aria-label={t('workflows.uploadWorkflow')}
        tooltip={t('workflows.uploadWorkflow')}
        icon={<PiUploadSimpleBold />}
        {...getRootProps()}
        pointerEvents="auto"
        variant="ghost"
      />

      <input {...getInputProps()} />
    </>
  );
};

export default memo(UploadWorkflowButton);
