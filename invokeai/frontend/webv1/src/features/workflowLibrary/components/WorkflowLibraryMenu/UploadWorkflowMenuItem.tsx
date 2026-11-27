import { MenuItem } from '@invoke-ai/ui-library';
import { useLoadWorkflowWithDialog } from 'features/workflowLibrary/components/LoadWorkflowConfirmationAlertDialog';
import { memo, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useTranslation } from 'react-i18next';
import { PiUploadSimpleBold } from 'react-icons/pi';

const UploadWorkflowMenuItem = () => {
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
