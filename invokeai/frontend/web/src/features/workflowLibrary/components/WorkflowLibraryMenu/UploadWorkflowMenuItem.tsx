import { MenuItem } from '@invoke-ai/ui-library';
import { useWorkflowLibraryModal } from 'features/nodes/store/workflowLibraryModal';
import { useLoadWorkflowFromFile } from 'features/workflowLibrary/hooks/useLoadWorkflowFromFile';
import { memo, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import { PiUploadSimpleBold } from 'react-icons/pi';

const UploadWorkflowMenuItem = () => {
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

  const { getRootProps, getInputProps } = useDropzone({
    accept: { 'application/json': ['.json'] },
    onDropAccepted,
    noDrag: true,
    multiple: false,
  });

  return (
    <MenuItem as="button" icon={<PiUploadSimpleBold />} {...getRootProps()}>
      {t('workflows.uploadWorkflow')}
      <input {...getInputProps()} />
    </MenuItem>
  );
};

export default memo(UploadWorkflowMenuItem);
