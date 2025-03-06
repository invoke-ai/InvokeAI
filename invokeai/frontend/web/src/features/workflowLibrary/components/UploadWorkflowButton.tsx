import { IconButton } from '@invoke-ai/ui-library';
import { useWorkflowListMenu } from 'features/nodes/store/workflowListMenu';
import { saveWorkflowAs } from 'features/workflowLibrary/components/SaveWorkflowAsDialog';
import { useLoadWorkflowFromFile } from 'features/workflowLibrary/hooks/useLoadWorkflowFromFile';
import { memo, useCallback, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import { PiUploadSimpleBold } from 'react-icons/pi';

const UploadWorkflowButton = () => {
  const { t } = useTranslation();
  const resetRef = useRef<() => void>(null);
  const workflowListMenu = useWorkflowListMenu();

  const loadWorkflowFromFile = useLoadWorkflowFromFile({
    resetRef,
    onSuccess: (workflow) => {
      workflowListMenu.close();
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
      <IconButton
        aria-label={t('workflows.uploadAndSaveWorkflow')}
        tooltip={t('workflows.uploadAndSaveWorkflow')}
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
